//
// Created by zhenyus on 1/16/19.
//

#include "lrb.h"
#include <algorithm>
#include "utils.h"
#include <chrono>
using namespace chrono;
using namespace std;
using namespace lrb;

void LRBCache::train() {
    ++n_retrain;
    auto timeBegin = chrono::system_clock::now();
    if (booster) LGBM_BoosterFree(booster);
    // create training dataset
    DatasetHandle trainData;
    LGBM_DatasetCreateFromCSR(
            static_cast<void *>(training_data->indptr.data()),
            C_API_DTYPE_INT32,
            training_data->indices.data(),
            static_cast<void *>(training_data->data.data()),
            C_API_DTYPE_FLOAT64,
            training_data->indptr.size(),
            training_data->data.size(),
            n_feature,  //remove future t
            training_params,
            nullptr,
            &trainData);

    LGBM_DatasetSetField(trainData,
                         "label",
                         static_cast<void *>(training_data->labels.data()),
                         training_data->labels.size(),
                         C_API_DTYPE_FLOAT32);

    // init booster
    LGBM_BoosterCreate(trainData, training_params, &booster);
    // train
    for (int i = 0; i < stoi(training_params["num_iterations"]); i++) {
        int isFinished;
        LGBM_BoosterUpdateOneIter(booster, &isFinished);
        if (isFinished) {
            break;
        }
    /*std::ofstream out( "predict.txt", std::ios::app);
    out << "*******************************************************************************************************************************" << "\n";
    out << "*******************************************************************************************************************************" << "\n";
    out << "*******************************************************************************************************************************" << "\n";
    out << "*******************************************************************************************************************************" << "\n";
    out << "*******************************************************************************************************************************" << "\n";
    out << "*******************************************************************************************************************************" << "\n";
    out << "*******************************************************************************************************************************" << "\n";
    out << "*******************************************************************************************************************************" << "\n";
    out << "*******************************************************************************************************************************" << "\n";
    out << "*****************************************************train is here**************************************************************" << "\n";

    out.close();*/
    }

    int64_t len;
    vector<double> result(training_data->indptr.size() - 1);
    LGBM_BoosterPredictForCSR(booster,
                              static_cast<void *>(training_data->indptr.data()),
                              C_API_DTYPE_INT32,
                              training_data->indices.data(),
                              static_cast<void *>(training_data->data.data()),
                              C_API_DTYPE_FLOAT64,
                              training_data->indptr.size(),
                              training_data->data.size(),
                              n_feature,  //remove future t
                              C_API_PREDICT_NORMAL,
                              0,
                              training_params,
                              &len,
                              result.data());


    double se = 0;
    for (int i = 0; i < result.size(); ++i) {
        auto diff = result[i] - training_data->labels[i];
        se += diff * diff;
    }
    training_loss = training_loss * 0.99 + se / batch_size * 0.01;

    LGBM_DatasetFree(trainData);
    training_time = 0.95 * training_time +
                    0.05 * chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - timeBegin).count();
}

void LRBCache::sample() {
    // start sampling once cache filled up
    auto rand_idx = _distribution(_generator);
    auto n_in = static_cast<uint32_t>(in_cache_metas.size());
    auto n_out = static_cast<uint32_t>(out_cache_metas.size());
    bernoulli_distribution distribution_from_in(static_cast<double>(n_in) / (n_in + n_out));
    auto is_from_in = distribution_from_in(_generator);
    if (is_from_in == true) {
        uint32_t pos = rand_idx % n_in;
        auto &meta = in_cache_metas[pos];
        meta.emplace_sample(current_seq);
    } else {
        uint32_t pos = rand_idx % n_out;
        auto &meta = out_cache_metas[pos];
        meta.emplace_sample(current_seq);
    }
}


void LRBCache::update_stat_periodic() {
//    uint64_t feature_overhead = 0;
//    uint64_t sample_overhead = 0;
//    for (auto &m: in_cache_metas) {
//        feature_overhead += m.feature_overhead();
//        sample_overhead += m.sample_overhead();
//    }
//    for (auto &m: out_cache_metas) {
//        feature_overhead += m.feature_overhead();
//        sample_overhead += m.sample_overhead();
//    }
    float percent_beyond;
    if (0 == obj_distribution[0] && 0 == obj_distribution[1]) {
        percent_beyond = 0;
    } else {
        percent_beyond = static_cast<float>(obj_distribution[1])/(obj_distribution[0] + obj_distribution[1]);
    }
    obj_distribution[0] = obj_distribution[1] = 0;
    segment_percent_beyond.emplace_back(percent_beyond);
    segment_n_retrain.emplace_back(n_retrain);
    segment_n_in.emplace_back(in_cache_metas.size());
    segment_n_out.emplace_back(out_cache_metas.size());

    float positive_example_ratio;
    if (0 == training_data_distribution[0] && 0 == training_data_distribution[1]) {
        positive_example_ratio = 0;
    } else {
        positive_example_ratio = static_cast<float>(training_data_distribution[1])/(training_data_distribution[0] + training_data_distribution[1]);
    }
    training_data_distribution[0] = training_data_distribution[1] = 0;
    segment_positive_example_ratio.emplace_back(positive_example_ratio);

    n_retrain = 0;
    cerr
            << "in/out metadata: " << in_cache_metas.size() << " / " << out_cache_metas.size() << endl
            //    cerr << "feature overhead: "<<feature_overhead<<endl;
            << "memory_window: " << memory_window << endl
//            << "percent_beyond: " << percent_beyond << endl
//            << "feature overhead per entry: " << static_cast<double>(feature_overhead) / key_map.size() << endl
            //    cerr << "sample overhead: "<<sample_overhead<<endl;
//            << "sample overhead per entry: " << static_cast<double>(sample_overhead) / key_map.size() << endl
            << "n_training: " << training_data->labels.size() << endl
            //            << "training loss: " << training_loss << endl
            << "training_time: " << training_time << " ms" << endl
            << "inference_time: " << inference_time << " us" << endl;
    assert(in_cache_metas.size() + out_cache_metas.size() == key_map.size());
}


bool LRBCache::lookup(SimpleRequest &req) {

    bool ret;
    ++current_seq;
    bool flag=false;
    int j =0;
    for(;j<ac.size();j++){
        if(ac[j]==req._id) {
            flag=true;
            break;
        }
    }
    if(flag==true)//有的情况下与队尾元素互换
    {
        for(;j<ac.size()-1;j++) ac[j]=ac[j+1];
        ac[ac.size()-1]=req._id;
    }else{
        while(ac.size()>=50){
            ac.pop_back();
        }
        ac.emplace_back(req._id);
    }
#ifdef EVICTION_LOGGING
    {
        AnnotatedRequest *_req = (AnnotatedRequest *) &req;
        auto it = future_timestamps.find(_req->_id);
        if (it == future_timestamps.end()) {
            future_timestamps.insert({_req->_id, _req->_next_seq});
        } else {
            it->second = _req->_next_seq;
        }
    }
#endif
    forget();
    /*if(current_seq>=200000&&current_seq%50==0&&current_seq<220000){//算50个结果
        predict_all();
        real_all();
    }
    if(current_seq==220000){
        int sum = 0;
        for(int q=0;q<change.size();q++){
            sum+=change[q];
            cout<<change[q]<<endl;
        }
        std::cerr<<sum/change.size()<<endl;
        getchar();
    }*/
    //first update the metadata: insert/update, which can trigger pending data.mature
    auto it = key_map.find(req._id);
    if (it != key_map.end()) {//所有window窗内request能在key map中找到
        auto list_idx = it->second.list_idx;
        auto list_pos = it->second.list_pos;
        Meta &meta = list_idx ? out_cache_metas[list_pos] : in_cache_metas[list_pos];
        //update past timestamps
        assert(meta._key == req._id);
        uint64_t last_timestamp = meta._past_timestamp;
        uint64_t forget_timestamp = last_timestamp % memory_window;
        //if the key in out_metadata, it must also in forget table
        assert((!list_idx) ||
               (negative_candidate_queue->find(forget_timestamp) !=
                negative_candidate_queue->end()));
        //re-request
        if (!meta._sample_times.empty()) {
            //mature
            for (auto &sample_time: meta._sample_times) {
                //don't use label within the first forget window because the data is not static
                uint32_t future_distance = current_seq - sample_time;
                training_data->emplace_back(meta, sample_time, future_distance, meta._key);
                ++training_data_distribution[1];
            }
            //batch_size ~>= batch_size
            if (training_data->labels.size() >= batch_size) {
                train();
                training_data->clear();
            }
            meta._sample_times.clear();
            meta._sample_times.shrink_to_fit();
        }

#ifdef EVICTION_LOGGING
        if (!meta._eviction_sample_times.empty()) {
            //mature
            for (auto &sample_time: meta._eviction_sample_times) {
                //don't use label within the first forget window because the data is not static
                uint32_t future_distance = req.seq - sample_time;
                eviction_training_data->emplace_back(meta, sample_time, future_distance, meta._key);
                //training
                if (eviction_training_data->labels.size() == batch_size) {
                    eviction_training_data->clear();
                }
            }
            meta._eviction_sample_times.clear();
            meta._eviction_sample_times.shrink_to_fit();
        }
#endif

        //make this update after update training, otherwise the last timestamp will change
#ifdef EVICTION_LOGGING
        AnnotatedRequest *_req = (AnnotatedRequest *) &req;
        meta.update(current_seq, _req->_next_seq);
#else
        meta.update(current_seq);//在这里对meta特征进行更新
#endif
        if (list_idx) {//如果请求meta在out cache
            negative_candidate_queue->erase(forget_timestamp);
            negative_candidate_queue->insert({current_seq % memory_window, req._id});
            assert(negative_candidate_queue->find(current_seq % memory_window) !=
                   negative_candidate_queue->end());
        } else {
            auto *p = dynamic_cast<InCacheMeta *>(&meta);
            p->p_last_request = in_cache_lru_queue.re_request(p->p_last_request);
        }
        //update negative_candidate_queue
        ret = !list_idx;
    } else {
        ret = false;
    }

    //sampling happens late to prevent immediate re-request
    if (is_sampling) {
        sample();
    }
    if(ret==true){
            std::ofstream out( "four_lru.txt", std::ios::app );
            out << req._t << " " << req._id << " " << req._size<<"           hit   " << "\n";
            out.close();
    }
    //如果命中了    
    //我们希望To_predict能保留最新的X个object，
    //所以每次有新的request到达时
    //如果它在out_cache中，admit部分会进行处理(如果没有压入，如果有更新信息且交换位置)
    //如果在in_cache里 admit部分不会进行处理，但需要在vector中判断有无以进行，如果有，则将该元素与队尾元素内容交换
    /*if(ret==true){
        bool that=false;
        int i =0;
        for(; i<To_predict.size();i++){
            if(To_predict[i].id == req._id) {
                that = true;//如果遍历可以找到
                break;
            }
        }//循环结束的时候如果that为true，fine会指向vector内需要修改的元素
        if(that == true) {
            //与队尾元素交换
            uint64_t tmp_id=To_predict[To_predict.size()-1].id;
            unsigned int tmp_idx=To_predict[To_predict.size()-1].value.list_idx;
            unsigned int tmp_pos=To_predict[To_predict.size()-1].value.list_pos;
            To_predict[To_predict.size()-1].id=To_predict[i].id;
            To_predict[To_predict.size()-1].value.list_idx=To_predict[i].value.list_idx;
            To_predict[To_predict.size()-1].value.list_pos=To_predict[i].value.list_pos;
            To_predict[i].id=tmp_id;
            To_predict[i].value.list_idx=tmp_idx;
            To_predict[i].value.list_pos= tmp_pos;
        }
    }*/
    return ret;
}


void LRBCache::forget() {
    /*
     * forget happens exactly after the beginning of each time, without doing any other operations. For example, an
     * object is request at time 0 with memory window = 5, and will be forgotten exactly at the start of time 5.
     * */
    //remove item from forget table, which is not going to be affect from update
    auto it = negative_candidate_queue->find(current_seq % memory_window);
    if (it != negative_candidate_queue->end()) {
        auto forget_key = it->second;
        auto pos = key_map.find(forget_key)->second.list_pos;
        // Forget only happens at list 1
        assert(key_map.find(forget_key)->second.list_idx);
//        auto pos = meta_it->second.list_pos;
//        bool meta_id = meta_it->second.list_idx;
        auto &meta = out_cache_metas[pos];//forget只会删除outcache的内容

        //timeout mature
        if (!meta._sample_times.empty()) {
            //mature
            //todo: potential to overfill
            uint32_t future_distance = memory_window * 2;
            for (auto &sample_time: meta._sample_times) {
                //don't use label within the first forget window because the data is not static
                training_data->emplace_back(meta, sample_time, future_distance, meta._key);
                ++training_data_distribution[0];
            }
            //batch_size ~>= batch_size
            if (training_data->labels.size() >= batch_size) {
                train();
                training_data->clear();
            }
            meta._sample_times.clear();
            meta._sample_times.shrink_to_fit();
        }

#ifdef EVICTION_LOGGING
        //timeout mature
        if (!meta._eviction_sample_times.empty()) {
            //mature
            //todo: potential to overfill
            uint32_t future_distance = memory_window * 2;
            for (auto &sample_time: meta._eviction_sample_times) {
                //don't use label within the first forget window because the data is not static
                eviction_training_data->emplace_back(meta, sample_time, future_distance, meta._key);
                //training
                if (eviction_training_data->labels.size() == batch_size) {
                    eviction_training_data->clear();
                }
            }
            meta._eviction_sample_times.clear();
            meta._eviction_sample_times.shrink_to_fit();
        }
#endif

        assert(meta._key == forget_key);
        remove_from_outcache_metas(meta, pos, forget_key);
    }
}
//如果 lookup的结果是hit显然不需要修改map
void LRBCache::admit(SimpleRequest &req) {
    //admit meta有可能从outcache进入到incache中
    const uint64_t &size = req._size;
    // object feasible to store?
    if (size > _cacheSize) {
        LOG("L", _cacheSize, req._id, size);
        return;
    }
    /*Mymap dict;
    dict.id = req._id;//可能需要押入的dict
    dict.value.list_idx = 0;//准入index一定为0*/
    auto it = key_map.find(req._id);
    if (it == key_map.end()) {//如果从来没有出现过
        //fresh insert
        key_map.insert({req._id, {0, (uint32_t) in_cache_metas.size()}});
        //dict.value.list_pos = (uint32_t) in_cache_metas.size();
        auto lru_it = in_cache_lru_queue.request(req._id);
#ifdef EVICTION_LOGGING
        AnnotatedRequest *_req = (AnnotatedRequest *) &req;
        in_cache_metas.emplace_back(req.id, req.size, current_seq, req.extra_features, _req->_next_seq, lru_it);
#else
        in_cache_metas.emplace_back(req._id, req._size, current_seq, req._extra_features, lru_it);//压入上一次出现时间
#endif
        _currentSize += size;
        //this must be a fresh insert
//        negative_candidate_queue.insert({(current_seq + memory_window)%memory_window, req.id});
        if (_currentSize <= _cacheSize)
            return;
    } else {
        //bring list 1 to list 0
        //first move meta data, then modify hash table
        uint32_t tail0_pos = in_cache_metas.size();
        auto &meta = out_cache_metas[it->second.list_pos];
        auto forget_timestamp = meta._past_timestamp % memory_window;
        negative_candidate_queue->erase(forget_timestamp);
        auto it_lru = in_cache_lru_queue.request(req._id);
        in_cache_metas.emplace_back(out_cache_metas[it->second.list_pos], it_lru);
        //out的pop处理
        uint32_t tail1_pos = out_cache_metas.size() - 1;
        if (it->second.list_pos != tail1_pos) {
            //swap tail
            out_cache_metas[it->second.list_pos] = out_cache_metas[tail1_pos];
            key_map.find(out_cache_metas[tail1_pos]._key)->second.list_pos = it->second.list_pos;
        }
        out_cache_metas.pop_back();
        //out的pop处理
        //dict.value.list_pos = (uint32_t) in_cache_metas.size();
        it->second = {0, tail0_pos};
        _currentSize += size;
    }
    if (_currentSize > _cacheSize) {//从第一次需要删除开始每次进行采样
        //start sampling once cache is filled up
        is_sampling = true;
    }
    // check more eviction needed?
    /*if (_currentSize <=_cacheSize) {
            std::ofstream out( "four_lru.txt", std::ios::app );
            out << req._t << " " << req._id << " " << req._size<<" !hit and no need to evict " << "\n";
            out.close();
    }*//*
        bool that=false;
        int i =0;
        for(; i<To_predict.size();i++){
            if(To_predict[i].id == req._id) {
                that = true;//如果遍历可以找到
                break;
            }
        }//循环结束的时候如果that为true，fine会指向vector内需要修改的元素
        if(that == false) {
            //它在To_predict中不存在，押入它的信息
            if(To_predict.size()<400){
                To_predict.emplace_back(dict);
            }else{
                To_predict.pop_back();
                To_predict.emplace_back(dict);
            }
        }else{
            //out->in
            To_predict[i].value.list_idx=0;
            To_predict[i].value.list_pos= dict.value.list_pos;
            //与队尾元素交换
            uint64_t tmp_id=To_predict[To_predict.size()-1].id;
            unsigned int tmp_idx=To_predict[To_predict.size()-1].value.list_idx;
            unsigned int tmp_pos=To_predict[To_predict.size()-1].value.list_pos;
            To_predict[To_predict.size()-1].id=To_predict[i].id;
            To_predict[To_predict.size()-1].value.list_idx=To_predict[i].value.list_idx;
            To_predict[To_predict.size()-1].value.list_pos=To_predict[i].value.list_pos;
            To_predict[i].id=tmp_id;
            To_predict[i].value.list_idx=tmp_idx;
            To_predict[i].value.list_pos= tmp_pos;
        }*/

    while (_currentSize > _cacheSize) {
        /*std::ofstream out( "four_lru.txt", std::ios::app );
        out << req._t << " " << req._id << " " << req._size<<" evict：   " << "\n";
        out.close();*/
        evict();//删除的内容如果To_predict中存在则需要更新，如果不存在不做修改
    }
}


pair<uint64_t, uint32_t> LRBCache::rank() {
    {
        //if not trained yet, or in_cache_lru past memory window, use LRU
        auto &candidate_key = in_cache_lru_queue.dq.back();//值引用
        //cout<<candidate_key<<endl;
        /*int32_t max_value=0;
        unordered_set<uint64_t> key_set;
        for(int i=0;i<1;){
            auto rand_idx = _distribution(_generator)%in_cache_lru_queue.dq.size();
            if(key_set.find(rand_idx) == key_set.end()) {
                i++;
                key_set.insert(rand_idx);
                if(rand_idx>max_value) max_value=rand_idx;
            }
        }//在这段

        
       list<int64_t>::const_iterator candidate_key;//拿到一个list<int> 类型的迭代器        //遍历list在min value处停止
        auto itor = in_cache_lru_queue.dq.begin();
        int i =0;
        while (itor != in_cache_lru_queue.dq.end()){
            if(i==max_value){
                candidate_key=itor;
                break;
            }else{
                *itor++;
                i++;
            }
        }
        */
        auto it = key_map.find(candidate_key);
        
       auto pos = it->second.list_pos;
        auto &meta = in_cache_metas[pos];
        if ((!booster) || (memory_window <= current_seq - meta._past_timestamp)) {//如果lru队列中存储的待删除对象过于老旧或或者没有模型则采用lru进行删除
            //this use LRU force eviction, consider sampled a beyond boundary object
            //如果该object超出boundary，删除该object
            if (booster) {
                ++obj_distribution[1];
            }
            return {meta._key, pos};
        }
    }

    //否则采用LRB进行删除，需要元素在这里初始化
    int32_t indptr[sample_rate + 1];
    indptr[0] = 0;
    int32_t indices[sample_rate * n_feature];
    double data[sample_rate * n_feature];
    int32_t past_timestamps[sample_rate];
    uint32_t sizes[sample_rate];

    unordered_set<uint64_t> key_set;
    uint64_t keys[sample_rate];
    uint32_t poses[sample_rate];
    //next_past_timestamp, next_size = next_indptr - 1

    unsigned int idx_feature = 0;
    unsigned int idx_row = 0;

    auto n_new_sample = sample_rate - idx_row;//需要新采样对象个数
    while (idx_row != sample_rate) {//这个while一直持续到对meta的操作结束
        uint32_t pos = _distribution(_generator) % in_cache_metas.size();//这里是有可能采到刚刚压入对象

        /*if(idx_row == 0){
            pos = in_cache_metas.size() -1;
        }//默认无准入*/
        auto &meta = in_cache_metas[pos];
        if (key_set.find(meta._key) != key_set.end()) {//如果该元素已经压入，重新采样
            continue;
        } else {
            key_set.insert(meta._key);
        }//采样部分 在采样开始之前把刚刚放进incache的meta放入


#ifdef EVICTION_LOGGING
        meta.emplace_eviction_sample(current_seq);
#endif

        keys[idx_row] = meta._key;
        poses[idx_row] = pos;
        //fill in past_interval
        indices[idx_feature] = 0;//开始为0
        data[idx_feature++] = current_seq - meta._past_timestamp;//这个地方需要调整
        past_timestamps[idx_row] = meta._past_timestamp;

        uint8_t j = 0;
        uint32_t this_past_distance = 0;
        uint8_t n_within = 0;
        if (meta._extra) {
            for (j = 0; j < meta._extra->_past_distance_idx && j < max_n_past_distances; ++j) {
                uint8_t past_distance_idx = (meta._extra->_past_distance_idx - 1 - j) % max_n_past_distances;
                uint32_t &past_distance = meta._extra->_past_distances[past_distance_idx];
                this_past_distance += past_distance;
                indices[idx_feature] = j+ 1;
                data[idx_feature++] = past_distance;
                if (this_past_distance < memory_window) {
                    ++n_within;
                }
//                } else
//                    break;
            }
        }

        indices[idx_feature] = max_n_past_timestamps-1;
        data[idx_feature++] = meta._size;
        sizes[idx_row] = meta._size;

        for (uint k = 0; k < n_extra_fields; ++k) {
            indices[idx_feature] = max_n_past_timestamps + k + 1;
            data[idx_feature++] = meta._extra_features[k];
        }

        indices[idx_feature] = max_n_past_timestamps + n_extra_fields + 1;
        data[idx_feature++] = n_within;

        for (uint8_t k = 0; k < n_edc_feature; ++k) {
            indices[idx_feature] = max_n_past_timestamps + n_extra_fields + 2 + k;
            uint32_t _distance_idx = min(uint32_t(current_seq - meta._past_timestamp) / edc_windows[k],
                                         max_hash_edc_idx);
            if (meta._extra)
                data[idx_feature++] = meta._extra->_edc[k] * hash_edc[_distance_idx];
            else
                data[idx_feature++] = hash_edc[_distance_idx];
        }
        //remove future t
        indptr[++idx_row] = idx_feature;
    }

    int64_t len;
    double scores[sample_rate];
    system_clock::time_point timeBegin;
    //sample to measure inference time
    if (!(current_seq % 10000))
        timeBegin = chrono::system_clock::now();
    LGBM_BoosterPredictForCSR(booster,
                              static_cast<void *>(indptr),
                              C_API_DTYPE_INT32,
                              indices,
                              static_cast<void *>(data),
                              C_API_DTYPE_FLOAT64,
                              idx_row + 1,
                              idx_feature,
                              n_feature,  //remove future t
                              C_API_PREDICT_NORMAL,
                              0,
                              inference_params,
                              &len,
                              scores);
    if (!(current_seq % 10000))
        inference_time = 0.95 * inference_time +
                         0.05 *
                         chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now() - timeBegin).count();
//    for (int i = 0; i < n_sample; ++i)
//        result[i] -= (t - past_timestamps[i]);//预测的就是下一次到达时间
    for (int i = sample_rate - n_new_sample; i < sample_rate; ++i) {
        //only monitor at the end of change interval
        if (scores[i] >= log1p(memory_window)) {
            ++obj_distribution[1];
        } else {
            ++obj_distribution[0];
        }
    }

    if (objective == object_miss_ratio) {
        for (uint32_t i = 0; i < sample_rate; ++i)
            scores[i] *= sizes[i];
    }

    vector<int> index(sample_rate, 0);
    for (int i = 0; i < index.size(); ++i) {
        index[i] = i;
    }

    sort(index.begin(), index.end(),
         [&](const int &a, const int &b) {
             return (scores[a] > scores[b]);
         }
    );

    //不妨在这里生成一个0-X的随机数
    //uint32_t pos_return = ( _distribution(_generator)) %2;

#ifdef EVICTION_LOGGING
    {
        if (start_train_logging) {
//            training_and_prediction_logic_timestamps.emplace_back(current_seq / 65536);
            for (int i = 0; i < sample_rate; ++i) {
                int current_idx = indptr[i];
                for (int p = 0; p < n_feature; ++p) {
                    if (p == indices[current_idx]) {
                        trainings_and_predictions.emplace_back(data[current_idx]);
                        if (current_idx + 1 < indptr[i + 1])
                            ++current_idx;
                    } else
                        trainings_and_predictions.emplace_back(NAN);
                }
                uint32_t future_interval = future_timestamps.find(keys[i])->second - current_seq;
                future_interval = min(2 * memory_window, future_interval);
                trainings_and_predictions.emplace_back(future_interval);
                trainings_and_predictions.emplace_back(result[i]);
                trainings_and_predictions.emplace_back(current_seq);
                trainings_and_predictions.emplace_back(1);
                trainings_and_predictions.emplace_back(keys[i]);
            }
        }
    }
#endif
    /*for(uint32_t i =0;i<sample_rate;i++)
     std::cerr<<scores[i]<<endl;*/
    return {keys[index[0]], poses[index[0]]};
}

void LRBCache::evict() {
    auto epair = rank();//将rank返回的结果删除
    uint64_t &key = epair.first;
    uint32_t &old_pos = epair.second;//这里返回了需要删除的内容

#ifdef EVICTION_LOGGING
    {
        auto it = future_timestamps.find(key);
        unsigned int decision_qulity =
                static_cast<double>(it->second - current_seq) / (_cacheSize * 1e6 / byte_million_req);
        decision_qulity = min((unsigned int) 255, decision_qulity);
        eviction_qualities.emplace_back(decision_qulity);
        eviction_logic_timestamps.emplace_back(current_seq / 65536);
    }
#endif

    auto &meta = in_cache_metas[old_pos];//要删除的

    /*std::ofstream out( "four_lru.txt", std::ios::app);
    out << meta._past_timestamp<< " " << meta._key << " " << meta._size<<"    NULL     " << "\n";
    out.close();*/
    if (memory_window <= current_seq - meta._past_timestamp) {//看需不需要扔到out_cache里
        //must be the tail of lru
        if (!meta._sample_times.empty()) {
            //mature
            uint32_t future_distance = current_seq - meta._past_timestamp + memory_window;
            for (auto &sample_time: meta._sample_times) {
                //don't use label within the first forget window because the data is not static
                training_data->emplace_back(meta, sample_time, future_distance, meta._key);
                ++training_data_distribution[0];
            }
            //batch_size ~>= batch_size
            if (training_data->labels.size() >= batch_size) {
                train();
                training_data->clear();
            }
            meta._sample_times.clear();
            meta._sample_times.shrink_to_fit();
        }

#ifdef EVICTION_LOGGING
        //must be the tail of lru
        if (!meta._eviction_sample_times.empty()) {
            //mature
            uint32_t future_distance = current_seq - meta._past_timestamp + memory_window;
            for (auto &sample_time: meta._eviction_sample_times) {
                //don't use label within the first forget window because the data is not static
                eviction_training_data->emplace_back(meta, sample_time, future_distance, meta._key);
                //training
                if (eviction_training_data->labels.size() == batch_size) {
                    eviction_training_data->clear();
                }
            }
            meta._eviction_sample_times.clear();
            meta._eviction_sample_times.shrink_to_fit();
        }
#endif

        in_cache_lru_queue.dq.erase(meta.p_last_request);
        meta.p_last_request = in_cache_lru_queue.dq.end();
        //above is suppose to be below, but to make sure the action is correct
//        in_cache_lru_queue.dq.pop_back();
        meta.free();
        _currentSize -= meta._size;
        key_map.erase(key);//超过memory_window删除
        uint32_t activate_tail_idx = in_cache_metas.size() - 1;
        if (old_pos != activate_tail_idx) {
            //move tail
            in_cache_metas[old_pos] = in_cache_metas[activate_tail_idx];
            key_map.find(in_cache_metas[activate_tail_idx]._key)->second.list_pos = old_pos;
        }//覆盖并修改key_map中的pos
        in_cache_metas.pop_back();
        ++n_force_eviction;
    } else {
        //bring list 0 to list 1 否则扔到out_cache里
        in_cache_lru_queue.dq.erase(meta.p_last_request);//维护lru队列

        meta.p_last_request = in_cache_lru_queue.dq.end();
        _currentSize -= meta._size;
        negative_candidate_queue->insert({meta._past_timestamp % memory_window, meta._key});

        uint32_t new_pos = out_cache_metas.size();
        out_cache_metas.emplace_back(in_cache_metas[old_pos]);
        uint32_t activate_tail_idx = in_cache_metas.size() - 1;
        if (old_pos != activate_tail_idx) {
            //move tail
            in_cache_metas[old_pos] = in_cache_metas[activate_tail_idx];
            key_map.find(in_cache_metas[activate_tail_idx]._key)->second.list_pos = old_pos;//key_map修改
        }//in_cache_meta的删除是将删除的object和vector尾部object交换然后删除，不会影响到尾部object以外内容的pos
        in_cache_metas.pop_back();
        key_map.find(key)->second = {1, new_pos};//key_map修改
        
        /*
        bool that=false;
        int q=0;
        //如果删除的时候找不到就不关To_predict的事情
        for(; q<To_predict.size();q++){
            if(To_predict[q].id == meta._key) {
                that = true;//如果遍历可以找到
                break;
            }
        }
        //找删除的
        if(that == true) {
            To_predict[q].value.list_idx=1;
            To_predict[q].value.list_pos= new_pos;//将需要删除的object信息修改
        }
        //找移位的
        q = 0;
        for(; q<To_predict.size();q++){
            if(To_predict[q].id == in_cache_metas[old_pos]._key) {
                that = true;//如果遍历可以找到
                break;
            }
        }//循环结束的时候如果that为true，fine会指向vector内需要修改的元素
        if(that==true){
               To_predict[q].value.list_pos= old_pos;
        }*/
    }
}

void LRBCache::remove_from_outcache_metas(Meta &meta, unsigned int &pos, const uint64_t &key) {
    //free the actual content
    //meta从out中被移除时候也需要调整To_predict的pos
    meta.free();
    //TODO: can add a function to delete from a queue with (key, pos)
    //evict
    uint32_t tail_pos = out_cache_metas.size() - 1;
    if (pos != tail_pos) {
        //swap tail
        out_cache_metas[pos] = out_cache_metas[tail_pos];
        key_map.find(out_cache_metas[tail_pos]._key)->second.list_pos = pos;
    }
    out_cache_metas.pop_back();
    key_map.erase(key);
    negative_candidate_queue->erase(current_seq % memory_window);
}

void LRBCache::predict_all(){
    //定义一个predict_all函数，记录key_map中所有meta的上一次到达时间和预测的下一次到达时间
    //我们还需要这些object实际的下一次到达时间进行比较，因此我们读取需要修改simulation.cpp，使其读取经过belady处理后的trace然后打印输出
    /*for(int q=0;q<key_map.size();q++){
        std::cerr<<"*********************************************************************"<<endl;
        std::cerr<<key_map[q].id<<"  **  "<<key_map[q].value.list_idx<<"  **  "<<To_predict[q].value.list_pos<<"  **  "<<endl;
    }*/
    int32_t indptr[in_cache_metas.size() + 1];
    indptr[0] = 0;
    int32_t indices[in_cache_metas.size() * n_feature];
    double data[in_cache_metas.size() * n_feature];
    int32_t past_timestamps[in_cache_metas.size()];
    uint32_t sizes[in_cache_metas.size()];

    unordered_set<uint64_t> key_set;
    uint64_t keys[in_cache_metas.size()];
    uint32_t poses[in_cache_metas.size()];
    uint32_t p_idx[in_cache_metas.size()];
    //next_past_timestamp, next_size = next_indptr - 1

    unsigned int idx_feature = 0;
    unsigned int idx_row = 0;

    uint32_t pos =0;//key_map时置1

    while (idx_row < in_cache_metas.size()) {//key_map.size()这个while一直持续到对meta的操作结束        
        uint32_t list_idx;
        uint32_t list_pos;
        //先在vector中查找有没有key值为pos的object
        auto it = key_map.find(in_cache_metas[pos]._key);
        if(it!=key_map.end()){
            list_idx = it->second.list_idx;
            list_pos = it->second.list_pos;
        }else{
            pos++;
            continue;
        }
        
        Meta &meta = list_idx ? out_cache_metas[list_pos] : in_cache_metas[list_pos];
        keys[idx_row] = meta._key;
        poses[idx_row] = list_pos;//记录key值和pos值
        p_idx[idx_row] = list_idx;
        //fill in past_interval
        indices[idx_feature] = 0;
        data[idx_feature++] = current_seq - meta._past_timestamp;//这个地方需要调整
        past_timestamps[idx_row] = meta._past_timestamp;

        uint8_t j = 0;
        uint32_t this_past_distance = 0;
        uint8_t n_within = 0;
        if (meta._extra) {
            for (j = 0; j < meta._extra->_past_distance_idx && j < max_n_past_distances; ++j) {
                uint8_t past_distance_idx = (meta._extra->_past_distance_idx - 1 - j) % max_n_past_distances;
                uint32_t &past_distance = meta._extra->_past_distances[past_distance_idx];
                this_past_distance += past_distance;
                indices[idx_feature] = j+1 ;
                data[idx_feature++] = past_distance;
                if (this_past_distance < memory_window) {
                    ++n_within;
                }
            }
        }

        indices[idx_feature] = max_n_past_timestamps;
        data[idx_feature++] = meta._size;
        sizes[idx_row] = meta._size;

        for (uint k = 0; k < n_extra_fields; ++k) {
            indices[idx_feature] = max_n_past_timestamps + k+1 ;
            data[idx_feature++] = meta._extra_features[k];
        }

        indices[idx_feature] = max_n_past_timestamps + n_extra_fields+1 ;
        data[idx_feature++] = n_within;

        for (uint8_t k = 0; k < n_edc_feature; ++k) {
            indices[idx_feature] = max_n_past_timestamps + n_extra_fields + 2 + k;
            uint32_t _distance_idx = min(uint32_t(current_seq - meta._past_timestamp) / edc_windows[k],
                                         max_hash_edc_idx);
            if (meta._extra)
                data[idx_feature++] = meta._extra->_edc[k] * hash_edc[_distance_idx];
            else
                data[idx_feature++] = hash_edc[_distance_idx];
        }
        //remove future t
        indptr[++idx_row] = idx_feature;
        pos++;
    }

    int64_t len;
    double scores[key_map.size() ];
    system_clock::time_point timeBegin;
    //sample to measure inference time
    if (!(current_seq % 10000))
        timeBegin = chrono::system_clock::now();
    LGBM_BoosterPredictForCSR(booster,
                              static_cast<void *>(indptr),
                              C_API_DTYPE_INT32,
                              indices,
                              static_cast<void *>(data),
                              C_API_DTYPE_FLOAT64,
                              idx_row + 1,
                              idx_feature,
                              n_feature,  //remove future t
                              C_API_PREDICT_NORMAL,
                              0,
                              inference_params,
                              &len,
                              scores);
    if (!(current_seq % 10000))
        inference_time = 0.95 * inference_time +
                         0.05 *
                         chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now() - timeBegin).count();
    //这里新建一个txt文件，命名为accuracy.txt，输出是key_map中所有object的信息
    
    idx_row = 0;
    std::ofstream out( "accuracy.txt");//, std::ios::app);
    while (idx_row < in_cache_metas.size()) {
        Meta &meta = p_idx[idx_row] ? out_cache_metas[poses[idx_row]] : in_cache_metas[poses[idx_row]];
        out << meta._key<< " " << meta._past_timestamp << " " <<scores[idx_row]<< "\n";
        idx_row++;
    }
    out.close();
}

void LRBCache::real_all(){
    //读取离线belady.ant数据
    ifstream infile("out.txt.ant");
    uint64_t id;
    int64_t i = 0; 
    //not actually need t and size
    std::ofstream out( "real.txt");//, std::ios::app);
    uint64_t next_t, t, size;
    uint64_t sequence=0;
    map<uint64_t, uint32_t> future_seen;//对所有object建立一个到下一次出现时间的映射
    map<uint64_t, uint32_t> now_seen;
    while(infile>>next_t >>t >> id >> size) {
        future_seen[id]=next_t;
        now_seen[id]=sequence;
        sequence++;
        if(sequence>current_seq) break;
    } 

    while(i<in_cache_metas.size()){
        //先看这个id在To_predict中有没有
        if(future_seen.find(in_cache_metas[i]._key)!=future_seen.end()){
            out << in_cache_metas[i]._key<< " " << now_seen[in_cache_metas[i]._key] << " " <<future_seen[in_cache_metas[i]._key]-current_seq << "\n";
            i++;
            //std::cerr<<i<<endl;
        }else{
            //std::cerr<<ac.size()<<" "<<ac[i]<<" "<<i<<endl;
        }
    }
    infile.close();
    out.close();
    estimate();
}

void LRBCache::estimate(){
    vector<int> index_real(ac.size(), 0);
    vector<int> index_predict(ac.size(), 0);
    for (int i = 0; i < ac.size(); ++i) {
        index_real[i] = i;
        index_predict[i] = i;
    }
    vector<int>distance_real;
    vector<int>distance_predict;
    ifstream infile_real("real.txt");
    uint64_t now, id, future;
    while(infile_real>>id >>now >> future) {
        distance_real.emplace_back(future);
    } 

    ifstream infile_predict("accuracy.txt");
    while(infile_predict>>id >>now >> future) {
        distance_predict.emplace_back(future);
    } 
    sort(index_real.begin(), index_real.end(),
         [&](const int &a, const int &b) {
             return (distance_real[a] > distance_real[b]);
         });

    sort(index_predict.begin(), index_predict.end(),
         [&](const int &a, const int &b) {
             return (distance_predict[a] > distance_predict[b]);
         } );   
    //从下面开始比较predict和real两个序列之间的相似度
    uint64_t result = comPare(index_real,index_predict);
    //std::cerr<<"***********************"<<result<<"***************************"<<endl;
    change.emplace_back(result);
}

uint64_t LRBCache::comPare(vector<int>index_real,vector<int>index_predict){
    map<int,int> real;
    map<int,int> predict;
    int target = index_predict[0];//预测的要删除object id
    for(int i =0;i<index_real.size();i++){
        real[index_real[i]]=i;
        predict[index_predict[i]]=i;
    }//得到id->下标映射
    //遍历计算位均差
    uint64_t loss=0;
    /*for(int i =0;i<index_real.size();i++){
        loss+=abs(real[i]-predict[i]);
        //std::cerr<<real[i]<<"***********"<<predict[i]<<"**********"<<index_real.size()<<endl;
    }//得到id->下标映射*/
    loss+=abs(real[target]);
    return loss;
}
//待完成
//固定一个大小为100的数据结构存储距离当前时刻最近的object
//对这部分object进行预测


//也可以对缓存内object进行预测，结果能更好的翻译预测性能
