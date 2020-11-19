# LRB Window Tuning

We set LRB memory window by using a small portion (first 20%) of the trace as development dataset. 
Below are the instructions to tune LRB memory window and run LRB with tuned window on the full trace.

The example is to run LRB on Wikipedia trace with 64/128/256/512/1024 GB cache sizes.

* Install LRB following the [instructions](INSTALL.md).
* Set up a mongodb instance. LRB uses this to store tuning results. 
* Set up the trace and machine to run. See [job_dev.yaml](config/job_dev.yaml) as an example.
Note [GNU parallel](https://www.gnu.org/software/parallel/) is used to running multiple tasks.
* Set up the cache sizes to run. See [trace_params.yaml](config/trace_params.yaml) as an example
* Run scripts. After running the best memory window results will be printed.
```shell script
python3 python-package/pywebcachesim/lrb_window_search.py ${YOUR JOB CONFIG FILE} ${YOUR ALGORITHM PARAM FILE} ${YOUR TRACE PARAM FILE} ${MONGODB URI}
```

