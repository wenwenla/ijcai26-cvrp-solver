#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <vector>

#include <iostream>
#include <thread>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <future>

class ThreadPool {
public:
    ThreadPool(size_t numThreads) : stop(false) {
        for (size_t i = 0; i < numThreads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(this->queueMutex);
                        this->condition.wait(lock, [this] {
                            return this->stop || !this->tasks.empty();
                        });

                        if (this->stop && this->tasks.empty()) {
                            return;
                        }

                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }

                    task();
                }
            });
        }
    }

    template<typename F, typename... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type> {
        using return_type = typename std::result_of<F(Args...)>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

        std::future<return_type> res = task->get_future();

        {
            std::unique_lock<std::mutex> lock(queueMutex);

            if (stop) {
                throw std::runtime_error("enqueue on stopped ThreadPool");
            }

            tasks.emplace([task]() { (*task)(); });
        }

        condition.notify_one();
        return res;
    }

    void join() {
        for (std::thread& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            stop = true;
        }
        condition.notify_all();
        join();
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;

    std::mutex queueMutex;
    std::condition_variable condition;
    bool stop;
};

namespace py = pybind11;

constexpr float EPS = 1e-6;


inline float sqr(float x) {
    return x * x;
}

inline float dist(const float* l, const float* r) {
    return sqrt(sqr(l[0] - r[0]) + sqr(l[1] - r[1]));
}


float solve(const float* data, const int* act, int n, bool o_flag, bool b_flag, bool l_flag, bool tw_flag) {
    // data[0]: x, 
    // data[1]: y, 
    // data[2]: demand, 
    // data[3]: ready time, 
    // data[4]: due time, 
    // data[5]: service time, 
    // data[6]: distance limit
    std::vector<float> p(n);
    p[0] = 0;
    for (int i = 1; i < n; ++i) {
        p[i] = 1e3;
    }
    float cost = 0;
    float arrival_time = 0;

    for (int t = 0; t < n - 1; ++t) {
    
        float loadout = 0, loadin = 0;
        int i = t + 1;

        bool backhaul_stage = false;
        while (i < n) {
            float this_demand = data[act[i - 1] * 7 + 2];
            
            if (this_demand < 0 - EPS) {
                backhaul_stage = true;
            }

            if (backhaul_stage && this_demand > 0 + EPS) {
                break; // violate the precede rule 
            }
            
            if (this_demand < 0 - EPS) {
                loadin -= this_demand;
            } else{
                loadout += this_demand;
            }
            
            if (loadin > 1 + EPS || loadout > 1 + EPS) {
                break; // violate capacity
            }

            if (i == t + 1) {
                cost = dist(&data[0], &data[act[i - 1] * 7]);
                arrival_time = dist(&data[0], &data[act[i - 1] * 7]);  // distance / speed
            } else {
                cost += dist(&data[act[i - 2] * 7], &data[act[i - 1] * 7]);
                arrival_time += dist(&data[act[i - 2] * 7], &data[act[i - 1] * 7]); // distance / speed;
            }


            if (tw_flag) {
                float ready_time = data[act[i - 1] * 7 + 3];
                float due_time = data[act[i - 1] * 7 + 4];
                float service_time = data[act[i - 1] * 7 + 5];
                
                if (arrival_time > due_time + EPS) {
                    break; // violate time window
                }

                if (arrival_time < ready_time) {
                    arrival_time = ready_time + service_time;
                } else {
                    arrival_time += service_time;
                }
            }
            
            float seg_cost = cost;
            if (!o_flag) {
                seg_cost += dist(&data[act[i - 1] * 7], &data[0]);
            }
            
            if (l_flag) {
                float distance_limit = data[0 * 7 + 6];
                if (seg_cost > distance_limit + EPS) {
                    // violate distance limit
                    break;
                }
            }

            float newval = p[t] + seg_cost;
            if (newval < p[i]) {
                p[i] = newval;
            }
            i = i + 1;
        }
    }

    return p[n - 1];
}


py::array_t<float> vrp_split(py::array_t<float> data, py::array_t<int> acts, std::vector<bool> flags) {

    bool o_flag = flags[0];
    bool b_flag = flags[1];
    bool l_flag = flags[2];
    bool tw_flag = flags[3];

    ThreadPool pool(100);
    auto data_buf = data.unchecked<3>();
    int batch_size = data_buf.shape(0);
    int n_nodes = data_buf.shape(1);

    auto data_ptr = static_cast<const float*>(data.data());
    auto acts_ptr = static_cast<const int*>(acts.data());

    auto result = py::array_t<float>(batch_size);
    auto out_ptr = static_cast<float*>(result.mutable_data());
    
    std::vector<std::shared_future<float>> futures;

    for (int i = 0; i < batch_size; ++i) {
        futures.push_back(
            pool.enqueue(solve, &data_ptr[i * n_nodes * 7], &acts_ptr[i * (n_nodes - 1)], n_nodes, o_flag, b_flag, l_flag, tw_flag)
        );
    }

    for (int i = 0; i < batch_size; ++i) {
        out_ptr[i] = futures[i].get();
    }
    return result;
}


PYBIND11_MODULE(gmsvrprf, m) {
    m.def("vrp_split", &vrp_split, "A function implemented in C++");
}