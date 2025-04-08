FROM ubuntu:jammy

EXPOSE 8001

# Copy only the necessary files for pip install
COPY ./requirement /app/requirement
COPY ./lib /app/lib

RUN apt update
RUN apt install python3 python3-pip -y

# apt-get换源并安装依赖
RUN sed -i "s@http://deb.debian.org@http://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list
RUN cat /etc/apt/sources.list
RUN apt-get update && apt-get install -y libgl1 libgomp1 libglib2.0-0 libsm6 libxrender1 libxext6
# 清理apt-get缓存
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

RUN dpkg -i app/lib/libssl1.1_1.1.0g-2ubuntu4_amd64.deb

WORKDIR /app

RUN pip3 install -r /app/requirement/core.txt
RUN pip3 install -r /app/requirement/requirements.txt
RUN pip3 install -r /app/requirement/sahi.txt

# CMD ["python3", "./main.py"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
