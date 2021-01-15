FROM ubuntu:18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
                           build-essential \
                           cmake \
                           git \
                           libboost-all-dev \
                           python3-pip \
                           python3-setuptools \
                           parallel \
                           libprocps-dev \
                           libssl-dev \
                           libsasl2-dev && \
                       apt-get update && apt-get install -y software-properties-common && \
                       add-apt-repository -y ppa:openjdk-r/ppa && apt-get update && \
                       apt-get install -y openjdk-8-jdk && \
                       rm -rf /var/lib/apt/lists/*

ENV WEBCACHESIM_ROOT=/opt/webcachesim
WORKDIR ${WEBCACHESIM_ROOT}

RUN git clone https://github.com/sunnyszy/lrb . && \
                       git submodule update --init --recursive && \
                       cd ${WEBCACHESIM_ROOT}/lib/LightGBM/build && \
                       cmake -DCMAKE_BUILD_TYPE=Release .. && \
                       make -j8 && \
                       make install && \
                       cd ${WEBCACHESIM_ROOT}/lib/mongo-c-driver/cmake-build/ && \
                       cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_AUTOMATIC_INIT_AND_CLEANUP=OFF .. && \
                       make -j8 && \
                       make install && \
                       cd ${WEBCACHESIM_ROOT}/lib/mongo-cxx-driver/build && \
                       cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local .. && \
                       make -j8 && \
                       make install && \
                       cd ${WEBCACHESIM_ROOT}/lib/libbf/build && \
                       cmake -DCMAKE_BUILD_TYPE=Release .. && \
                       make -j8 && \
                       make install && \
                       cd ${WEBCACHESIM_ROOT}/build && \
                       cmake -DCMAKE_BUILD_TYPE=Release .. && \
                       make -j8 && \
                       make install && \
                       ldconfig

ENV WEBCACHESIM_TRACE_DIR=/trace
ENTRYPOINT ["/opt/webcachesim/build/bin/webcachesim_cli"]
