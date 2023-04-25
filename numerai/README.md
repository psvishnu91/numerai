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

## Setting Lambdalabs for training
### Commands
local mc
``` bash
scp ~/.aws/personal_credentials ubunut@<ip>:~/
```
Remote mc
``` bash
mkdir ~/.aws/ 
mv ~/personal_credentials ~/.aws/credentials 
sudo jupyter notebook --generate-config
echo "c.NotebookApp.ip = '*'" >> /root/.jupyter/jupyter_notebook_config.py
git clone https://github.com/vispz/numerai.git
cd numerai/numerai 
sudo pip install -r requirements-minimal.txt
sudo pip install -U jupyter
cd ~
```

In a tmux shell start a notebook
``` bash
tmux
jupyter notebook
```
Open up Jupyter at IP:8888/?<token>.

