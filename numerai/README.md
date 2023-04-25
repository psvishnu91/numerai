# Deploying
## Creating requirements.txt file
```bash
virtualenv --python=python3.10 .venv && \
    source .venv/bin/activate && \
    pip insall -r requirments-minimal.txt && \
    pip freeze > deploy/requirements.txt
```

# Modelling
## Dockerising environment for modelling
``` bash
docker build  --platform linux/amd64 -f Dockerfile -t vishnups/numerai-visp .
  && docker push vishnups/numerai-visp
```
## Setting up aws
Spin up a `m6a.4xlarge` with ubuntu. Set up security group like below \
![](https://pasteboard.co/6DQvNjbMBTML.png?o=1)

In local m/c
``` bash
# example ubuntu@ec2-3-141-40-156.us-east-2.compute.amazonaws.com
IP=ubuntu@<ip>
PEM_FILE=
scp -i $PEM_FILE  ~/.aws/personal_credentials $IP:~/credentials
ssh -i $PEM_FILE $IP
```

Either run the following as is or add it to `setup.sh` and run `bash -x setup.sh`.
``` bash
sudo apt-get update
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y awscli docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl start docker
# Move credentials uploaded locally
mkdir -p ~/.aws/
cd ~
cp credentials ~/.aws/credentials
```


## Setting Lambdalabs for training
### Commands
local mc
``` bash
IP=
scp ~/.aws/personal_credentials ubuntu@${IP}:~/
ssh -A ubuntu@$IP
```
Remote mc
``` bash
mkdir ~/.aws/ 
mv ~/personal_credentials ~/.aws/credentials 
jupyter notebook --generate-config
echo "c.NotebookApp.ip = '*'" >> /home/ubuntu/.jupyter/jupyter_notebook_config.py
git clone https://github.com/vispz/numerai.git
cd numerai/numerai 
pip install -r requirements-minimal.txt
pip install -U jupyter
cd ~
```

In a tmux shell start a notebook
``` bash
tmux
jupyter notebook
```
Open up Jupyter at IP:8888/?<token>.

