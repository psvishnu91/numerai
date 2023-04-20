### Commands
``` bash
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