# Deploying
## Creating requirements.txt file
```bash
virtualenv --python=python3.10 .venv && \
    source .venv/bin/activate && \
    pip insall -r requirments-minimal.txt && \
    pip freeze > deploy/requirements.txt
```
## Deploying a new model
* Use the `task6-create-deploy-models.ipynb` notebook to create and pickle the new model.
* The pickled model needs to be saved under `models/` dir. Upload it to
  `s3://numerai-v1/deployed_models/` for posterity.
* The following command should run
  `predict_script.predict(napi=napi, wrapped_model=new_model)` and return
  a dataframe with one column `prediction` which has a ranked pct output.
* Update `predict.sh` to run `predict.py` once more for your new model.

# Streaming logs from remote m/c
``` bash
# in local m/c
ssh ubuntu@165.1.64.156  "tail -f numerai/numerai/log.txt"
```

# Modelling
## ## Option 1: [Lambdalabs] | Setting Lambdalabs for training (Cheaper)
### Commands
local mc
``` bash
# Example: 165.1.65.156
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


## Option 2: [EC2] | Setting up aws EC2 instance for model training

### Step 1: Docker build
In local m/c
``` bash
docker build  --platform linux/amd64 -f Dockerfile -t vishnups/numerai-visp .
  && docker push vishnups/numerai-visp
```

### Step 2: Spin up EC2 instance and add aws creds
Spin up a `m6a.4xlarge` with ubuntu. Set up security group like below \
<img src="https://gcdnb.pbrd.co/images/6DQvNjbMBTML.png?o=1" width="30%"/>

``` bash
# example ubuntu@ec2-3-141-40-156.us-east-2.compute.amazonaws.com
IP=ubuntu@<ip>
PEM_FILE=
scp -i $PEM_FILE  ~/.aws/personal_credentials $IP:~/credentials
ssh -i $PEM_FILE $IP
```

### Step 3: Setup the EC2 instance by install docker
Either run the following as is or create a `setup.sh` script and run `bash -x setup.sh`.
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

### Step 4: Docker run, modelling container
In a tmux shell in EC2
``` bash
tmux
PORT=8888 && sudo docker run --interactive -t \
    -v ~/.aws/credentials:/numerai/.aws/credentials \
    -v ~/data:/numerai/data/ \
    --publish "${PORT}":"${PORT}" \
    --expose="${PORT}" \
    vishnups/numerai-visp
jupyter notebook --allow-root
```
#### Step 5: Check the browser
Ensure you pick up the public IP and go to `http://<ip>:8888?token=<token>` \
<img src="https://gcdnb.pbrd.co/images/YbRxsBXjhD7D.png?o=1" width="50%"/>
