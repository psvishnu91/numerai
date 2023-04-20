### Commands
local mc
``` bash
scp ~/.aws/personal_credentials <id>:~/
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
cd ~
```

In a tmux shell start a notebook
``` bash
tmux
jupyter notebook
```
Open up Jupyter at IP:8888/?<token>.