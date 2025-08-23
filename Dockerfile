# CPU image with OpenFace + your Flask app
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# ---- System deps ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    git cmake build-essential pkg-config \
    ffmpeg libsndfile1 libgomp1 \
    libopenblas-dev liblapack-dev libeigen3-dev \
    libboost-system-dev libboost-filesystem-dev libboost-serialization-dev \
    libprotobuf-dev protobuf-compiler libgoogle-glog-dev libgflags-dev zlib1g-dev \
    libopencv-dev \
    libx11-dev libpng-dev libjpeg-turbo8-dev libtiff5-dev \
    libgtk2.0-0 libqt5core5a libqt5gui5 libqt5widgets5 \
    ca-certificates wget unzip \
 && rm -rf /var/lib/apt/lists/*

# Ensure /usr/local/lib is in runtime linker path
RUN sh -c 'echo /usr/local/lib > /etc/ld.so.conf.d/local.conf' && ldconfig

# ---- Build dlib (>=19.13 required by OpenFace) ----
RUN git clone --depth 1 --branch v19.24 https://github.com/davisking/dlib.git /opt/dlib && \
    cd /opt/dlib && mkdir build && cd build && \
    cmake .. -DBUILD_SHARED_LIBS=ON -DUSE_AVX_INSTRUCTIONS=ON && \
    cmake --build . --config Release -- -j"$(nproc)" && \
    cmake --install . && ldconfig

# ---- Build OpenFace against the new dlib ----
RUN git clone --depth 1 https://github.com/TadasBaltrusaitis/OpenFace.git /opt/OpenFace && \
    cd /opt/OpenFace && bash ./download_models.sh && \
    mkdir -p build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -Ddlib_DIR=/usr/local/lib/cmake/dlib .. && \
    make -j"$(nproc)" && make install && \
    ln -sf /usr/local/bin/FeatureExtraction /usr/bin/FeatureExtraction

ENV OPENFACE_EXE=/usr/local/bin/FeatureExtraction
ENV PATH="/usr/local/bin:${PATH}"

# ---- Python deps ----
COPY requirements.txt .
# scikit-learn pin keeps compatibility with your saved scaler
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir scikit-learn==1.6.1 && \
    python3 -m pip install --no-cache-dir -r requirements.txt

# ---- App ----
COPY . .
EXPOSE 7860
CMD ["python3", "app/main.py"]
