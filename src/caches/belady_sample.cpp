//
// Created by zhenyus on 12/17/18.
//

#include "belady_sample.h"
#include "utils.h"

using namespace std;


bool BeladySampleCache::lookup(SimpleRequest &_req) {
    auto & req = dynamic_cast<AnnotatedRequest &>(_req);
    current_t = req._t;
    auto it = key_map.find(req._id);
    if (it != key_map.end()) {//key_map中记录 的是id->pos的映射
        //update past timestamps
        uint32_t &pos_idx = it->second;
        meta_holder[pos_idx].update(req._t, req._next_seq);

        if (memorize_sample && memorize_sample_keys.find(req._id) != memorize_sample_keys.end() &&
            req._next_seq - current_t <= threshold)
            memorize_sample_keys.erase(req._id);

        std::cout<<"it is a hit"<<endl;
        return true;
    }
    std::cerr<<"it  is a miss"<<endl;
    return false;
}

void BeladySampleCache::admit(SimpleRequest &_req) {
    AnnotatedRequest & req = static_cast<AnnotatedRequest &>(_req);
    const uint64_t & size = req._size;

    // object feasible to store?s
    if (size > _cacheSize) {
        LOG("L", _cacheSize, req.get_id(), size);
        return;
    }

    auto it = key_map.find(req._id);
        //fresh insert
    key_map.insert({req._id, (uint32_t) meta_holder.size()});
    meta_holder.emplace_back(req._id, req._size, req._t, req._next_seq);

    _currentSize += size;
    // check more eviction needed?

    while (_currentSize > _cacheSize) {
        evict();
    }
}


pair<uint64_t, uint32_t> BeladySampleCache::rank() {
    vector<pair<uint64_t, uint32_t >> beyond_boundary_key_pos;
    uint64_t max_future_interval = 0;
    uint64_t max_key;
    uint32_t max_pos;

    if (memorize_sample) {
        //first pass: move near objects out of the set.
        for (auto it = memorize_sample_keys.cbegin(); it != memorize_sample_keys.end();) {
            auto &key = *it;
            auto &pos = key_map.find(key)->second;
            auto &meta = meta_holder[pos];
            uint64_t &past_timestamp = meta._past_timestamp;
            if (meta._future_timestamp - current_t <= threshold) {
                it = memorize_sample_keys.erase(it);//更新memorize，memorize记录的是cache里下一次到达时间超出boundary的object
            } else {
                beyond_boundary_key_pos.emplace_back(pair(key, pos));
                ++it;
            }
        }
    }

    uint n_sample = min(sample_rate, (uint32_t) meta_holder.size());
    //把这n_sample个全部记录下来，最后sort，尾部输出
    std::vector<pair<BeladySampleMeta,u_int64_t>> to_evict;
    std::vector<u_int32_t> to_pose; 
    for (uint32_t i = 0; i < n_sample; i++) {
        //true random sample
        uint32_t pos = (i + _distribution(_generator)) % meta_holder.size();//这里是采样结束
        auto &meta = meta_holder[pos];

        if (memorize_sample && memorize_sample_keys.find(meta._key) != memorize_sample_keys.end()) {
            //this key is already in the memorize keys, so we will enumerate it
            continue;
        }

        uint64_t future_interval=meta._future_timestamp - current_t;
        to_evict.emplace_back(make_pair(meta,future_interval));
        to_pose.emplace_back(pos);

        if (meta._future_timestamp - current_t <= threshold) {
            future_interval = meta._future_timestamp - current_t;
        } else {
            beyond_boundary_key_pos.emplace_back(pair(meta._key, pos));
            if (memorize_sample && memorize_sample_keys.size() < sample_rate) {
                memorize_sample_keys.insert(meta._key);
            }
            continue;
        }

        //select the first one: random one
        if (future_interval > max_future_interval) {
            max_future_interval = future_interval;
            max_key = meta._key;
            max_pos = pos;
        }
    }
    sort(to_evict.begin(), to_evict.end(),
         [&](const pair<BeladySampleMeta,u_int64_t> &a, const pair<BeladySampleMeta,u_int64_t> &b) {
             return (a.second > b.second);
         });//将meta future_interval组成map按value排序
    
    uint32_t pos_return = ( _distribution(_generator)) %64;//这里是采样结束
    if(pos_return>n_sample-1) pos_return=n_sample-1; 
    auto a=key_map.find(to_evict[pos_return].first._key);

    return {a->first, a->second};
  
    if (beyond_boundary_key_pos.empty()) {
        return {max_key, max_pos};
    } else {
        auto rand_id = _distribution(_generator) % beyond_boundary_key_pos.size();
        auto &item = beyond_boundary_key_pos[rand_id];
        return {item.first, item.second};
    }
}

void BeladySampleCache::evict() {
//    static uint counter = 0;
    auto epair = rank();
    uint64_t & key = epair.first;
    uint32_t & old_pos = epair.second;

    //record meta's future interval

#ifdef EVICTION_LOGGING
    {
        auto &meta = meta_holder[old_pos];
        //record eviction decision quality
        unsigned int decision_qulity =
                static_cast<double>(meta._future_timestamp - current_t) / (_cacheSize * 1e6 / byte_million_req);
        decision_qulity = min((unsigned int) 255, decision_qulity);
        eviction_distances.emplace_back(decision_qulity);
    }
#endif

    if (memorize_sample && memorize_sample_keys.find(key) != memorize_sample_keys.end())
        memorize_sample_keys.erase(key);

    _currentSize -= meta_holder[old_pos]._size;
    uint32_t activate_tail_idx = meta_holder.size() - 1;
    if (old_pos !=  activate_tail_idx) {
        //move tail
        meta_holder[old_pos] = meta_holder[activate_tail_idx];
        key_map.find(meta_holder[activate_tail_idx]._key)->second = old_pos;
    }
    meta_holder.pop_back();

    key_map.erase(key);
}