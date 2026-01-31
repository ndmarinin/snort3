//--------------------------------------------------------------------------
// Copyright (C) 2023-2025 Cisco and/or its affiliates. All rights reserved.
//
// This program is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License Version 2 as published
// by the Free Software Foundation.  You may not use, modify or distribute
// this program under any other version of the GNU General Public License.
//
// This program is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// General Public License for more details.
//
// You should have received a copy of the GNU General Public License along
// with this program; if not, write to the Free Software Foundation, Inc.,
// 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
//--------------------------------------------------------------------------
// snort_ml_engine.h author Vitalii Horbatov <vhorbato@cisco.com>
//                   author Brandon Stultz <brastult@cisco.com>

#ifndef SNORT_ML_ENGINE_H
#define SNORT_ML_ENGINE_H

#ifdef HAVE_LIBML
#include <libml.h>
#endif

#include <memory>
#include <unordered_map>
#include <utility>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

#include "framework/inspector.h"
#include "framework/module.h"
#include "hash/lru_cache_local.h"
#include "search_engines/search_tool.h"

#define SNORT_ML_ENGINE_NAME "snort_ml_engine"
#define SNORT_ML_ENGINE_HELP "configure machine learning engine settings"

// Mock BinaryClassifierSet for tests if LibML is absent
#ifndef HAVE_LIBML
namespace libml
{

class BinaryClassifierSet
{
public:
    bool build(const std::vector<std::string>& models)
    {
        if (!models.empty())
            pattern = models[0];

        return pattern != "error";
    }

    bool run(const char* ptr, size_t len, float& out)
    {
        std::string data(ptr, len);
        out = data.find(pattern) == std::string::npos ? 0.0f : 1.0f;
        return pattern != "fail";
    }

private:
    std::string pattern;
};

}
#endif

struct SnortMLEngineStats : public LruCacheLocalStats
{
    PegCount filter_searches;
    PegCount filter_matches;
    PegCount filter_allows;
    PegCount libml_calls;
};

// Forward declarations
class SnortMLEngine;
struct MLInferenceRequest;

// ============================================================================
// Optimized Batch Processing for DNS/ML Inference
// ============================================================================

// Single inference request
struct MLInferenceRequest
{
    const char* data;
    size_t length;
    uint64_t hash;
    float* result;
    std::atomic<bool>* done = nullptr;
    int request_id = 0;
};

// Thread-safe batch queue for async processing (forward declaration - impl below)
class MLBatchProcessor;

typedef LruCacheLocal<uint64_t, float, std::hash<uint64_t>> SnortMLCache;
typedef std::unordered_map<std::string, bool> SnortMLFilterMap;;

struct SnortMLContext
{
    libml::BinaryClassifierSet classifiers;
    std::unique_ptr<SnortMLCache> cache;
    std::unique_ptr<MLBatchProcessor> batch_processor;  // ✅ Async batch processing
    
    // ✅ OPTIMIZATION: Keep model buffers in memory (never reload)
    // Models are loaded once at configure() and kept resident in memory
    // This eliminates model load overhead on each inference
    std::vector<std::string> model_buffers;  // Raw TFLite model data
    std::atomic<size_t> total_model_size{0}; // For monitoring
};

struct SnortMLEngineConfig
{
    std::string http_param_model_path;
    std::vector<std::string> http_param_models;
    SnortMLFilterMap http_param_filters;

    bool has_allow = false;
    size_t cache_memcap = 0;
};

struct SnortMLSearch
{
    bool match = false;
    bool allow = false;
    bool has_allow = false;
};

class SnortMLEngineModule : public snort::Module
{
public:
    SnortMLEngineModule();

    bool begin(const char*, int, snort::SnortConfig*) override;
    bool set(const char*, snort::Value&, snort::SnortConfig*) override;

    const PegInfo* get_pegs() const override;
    PegCount* get_counts() const override;

    Usage get_usage() const override
    { return GLOBAL; }

    SnortMLEngineConfig get_config()
    {
        SnortMLEngineConfig out;
        std::swap(conf, out);
        return out;
    }

private:
    SnortMLEngineConfig conf;
};

class SnortMLEngine : public snort::Inspector
{
public:
    SnortMLEngine(SnortMLEngineConfig c) : conf(std::move(c)) {}
    ~SnortMLEngine() override
    { delete mpse; }

    bool configure(snort::SnortConfig*) override;
    void show(const snort::SnortConfig*) const override;

    void tinit() override;
    void tterm() override;

    void install_reload_handler(snort::SnortConfig*) override;

    bool scan(const char*, const size_t, float&) const;

private:
    bool read_models();
    bool read_model(const std::string&);

    SnortMLEngineConfig conf;
    snort::SearchTool* mpse = nullptr;
};

// ============================================================================
// MLBatchProcessor Implementation (after SnortMLEngine definition)
// ============================================================================

class MLBatchProcessor
{
public:
    MLBatchProcessor(size_t batch_size = 32, size_t max_queue = 1000)
        : batch_size(batch_size), max_queue_size(max_queue), 
          running(false), processed_count(0) {}

    ~MLBatchProcessor()
    {
        stop();
    }

    // Start background thread for batch processing
    void start(const SnortMLEngine* eng)
    {
        if (running.exchange(true))
            return;
        
        engine = eng;
        worker_thread = std::thread(&MLBatchProcessor::process_batches, this);
    }

    // Stop background thread
    void stop()
    {
        if (!running.exchange(false))
            return;
        
        cv.notify_all();
        if (worker_thread.joinable())
            worker_thread.join();
    }

    // Add inference request to queue (non-blocking)
    bool queue_request(const MLInferenceRequest& req)
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        if (request_queue.size() >= max_queue_size)
            return false;
        
        request_queue.push(req);
        if (request_queue.size() >= batch_size)
            cv.notify_one();
        
        return true;
    }

    // Synchronous inference (for immediate results)
    bool process_now(const char* data, size_t len, float& out)
    {
        if (!engine)
            return false;
        return engine->scan(data, len, out);
    }

    uint64_t get_processed_count() const
    {
        return processed_count;
    }

private:
    void process_batches()
    {
        std::vector<MLInferenceRequest> batch;
        batch.reserve(batch_size);

        while (running)
        {
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                cv.wait_for(lock, std::chrono::milliseconds(10), 
                           [this] { return request_queue.size() >= batch_size || !running; });

                // Drain queue into batch
                while (!request_queue.empty() && batch.size() < batch_size)
                {
                    batch.push_back(request_queue.front());
                    request_queue.pop();
                }
            }

            // Process batch (without holding lock)
            if (!batch.empty() && engine)
            {
                for (auto& req : batch)
                {
                    engine->scan(req.data, req.length, *req.result);
                    if (req.done)
                        req.done->store(true, std::memory_order_release);
                    processed_count++;
                }
                batch.clear();
            }
        }
    }

    const SnortMLEngine* engine = nullptr;
    size_t batch_size;
    size_t max_queue_size;
    
    std::queue<MLInferenceRequest> request_queue;
    std::mutex queue_mutex;
    std::condition_variable cv;
    std::atomic<bool> running;
    std::thread worker_thread;
    std::atomic<uint64_t> processed_count;
};

#endif
