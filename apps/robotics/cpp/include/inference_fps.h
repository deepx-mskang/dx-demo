#pragma once

#include <chrono>
#include <cstdio>
#include <deque>
#include <mutex>
#include <string>

class SlidingWindowInferenceFps
{
public:
    SlidingWindowInferenceFps()
        : last_ui_tick_(std::chrono::steady_clock::now())
    {
    }

    void RecordInferenceComplete()
    {
        const auto now = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lock(mutex_);
        prune_events_locked(now);
        events_.push_back(now);
    }

    const std::string &TickUi()
    {
        const auto now = std::chrono::steady_clock::now();
        if (now - last_ui_tick_ < std::chrono::milliseconds(200))
        {
            return label_;
        }
        last_ui_tick_ = now;

        size_t count = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            prune_events_locked(now);
            count = events_.size();
        }

        const double rate = static_cast<double>(count) / 5.0;
        char buf[48];
        std::snprintf(buf, sizeof(buf), "%.1f inf/s", rate);
        label_ = buf;
        return label_;
    }

private:
    void prune_events_locked(std::chrono::steady_clock::time_point now)
    {
        const auto cutoff = now - std::chrono::seconds(5);
        while (!events_.empty() && events_.front() < cutoff)
        {
            events_.pop_front();
        }
    }

    std::mutex mutex_;
    std::deque<std::chrono::steady_clock::time_point> events_;
    std::chrono::steady_clock::time_point last_ui_tick_;
    std::string label_ = "0.0 inf/s";
};
