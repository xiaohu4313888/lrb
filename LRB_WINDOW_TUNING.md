# LRB Window Tuning

We set LRB memory window by using a small portion (first 20%) of the trace as development dataset. 
Below are the instructions to tune LRB memory window and run LRB with tuned window on the full trace.

The example is to run LRB on Wikipedia trace with 64/128/256/512/1024 GB cache sizes.

* Install LRB following the [instructions](INSTALL.md).
* Set up a mongodb instance. LRB uses this to store tuning results. 
* Set up the trace and machine to run. See [job_lrb_window_search.yaml](config/job_lrb_window_search.yaml) as an example.
Note [GNU parallel](https://www.gnu.org/software/parallel/) is used to running multiple tasks.
* Set up the cache sizes to run. See [trace_params_lrb_window_search.yaml](config/trace_params_lrb_window_search.yaml) as an example
* run scripts. After running the LRB with best memory window results will be stored in the mongodb instance in dev collection. 
```shell script
python3 pywebcachesim/lrb_auto_search_memory_window.py --job_file config/job_lrb_window_search.yaml --algorithm_param_file config/algorithm_params_lrb_window_search.yaml --trace_param_file config/trace_params_lrb_window_search.yaml --dburi ${MONGODB URI}
```

