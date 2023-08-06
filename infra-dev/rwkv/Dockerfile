FROM ubuntu:22.04

RUN apt update -y
RUN apt install -y git curl xz-utils

# Install Zig and Zig language server. See:
#   https://ziglang.org/download/
#   https://install.zigtools.org/
ARG ZIG_PATH=/usr/local/zig
RUN PLATFORM=`uname -m` \
    && curl https://ziglang.org/builds/zig-linux-${PLATFORM}-0.11.0.tar.xz -o /tmp/zig.tar.xz \
    && cd /tmp/ \
    && tar -xf zig.tar.xz \
    && mv /tmp/zig-linux-${PLATFORM}-0.11.0 ${ZIG_PATH} \
    && rm /tmp/zig.tar.xz \
    && curl https://zig.pm/zls/downloads/${PLATFORM}-linux/bin/zls -o /usr/local/bin/zls \
    && chmod +x /usr/local/bin/zls
ENV PATH="$PATH:${ZIG_PATH}"

# Install python and packages
RUN apt install -y python3 python3-pip
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt
