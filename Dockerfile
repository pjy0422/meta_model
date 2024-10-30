FROM python:3.11-slim

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

WORKDIR / 

COPY members ./

RUN groupadd -g 1015 intern \
&& cat members >> /etc/passwd \
&& apt update \
&& apt install -yy -q wget htop build-essential checkinstall libbz2-dev zlib1g-dev libssl-dev liblzma-dev tmux software-properties-common \
&& apt install -yy -q python3.11-dev git \
&& apt-get update \
&& apt-get install -yy -q python3.11-venv zsh libgl1-mesa-glx tzdata gdb sudo vim curl \
&& python3.11 -m pip install --upgrade pip \
&& python3.11 -m pip install tqdm seaborn matplotlib lightgbm mlflow pandas numpy==1.26.4 scikit-learn pymfe xgboost keras tensorflow xlrd openpyxl

WORKDIR /home/workspace

# oh-my-zsh 설치
RUN ZSH="/home/.oh-my-zsh" sh -c "$(curl -fsSL https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)" "" --skip-chsh
# powerlevel10k theme 설치
RUN git clone --depth=1 https://github.com/romkatv/powerlevel10k.git /home/.oh-my-zsh/custom/themes/powerlevel10k
# autosuggestion 설치
RUN git clone https://github.com/zsh-users/zsh-autosuggestions /home/.oh-my-zsh/custom/plugins/zsh-autosuggestions
RUN git clone https://github.com/zsh-users/zsh-syntax-highlighting.git /home/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting

COPY .zshrc /home/
RUN chmod -R 777 /home \
&& chgrp -R intern /home 

RUN apt-get clean && \
    apt-get autoclean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/cache/* && \
    rm -rf /var/lib/log/* && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /var/cache/apt/archives/*