# - CPU
# ```docker run -Pit --privileged --name dlhk --rm -v $HOME:/tf/hyunkim -e "0000" -p 8810:8888 -p 6060:6006 tensorflow/tensorflow:latest-py3-jupyter```

# - GPU
# ```docker run -Pit --privileged --name dlhk --runtime=nvidia -v /home/hyunkim:/home/hyunkim -e "0000" -p 8810:8888 -p 6060:6006 tensorflow/tensorflow:latest-gpu-py3```

# - Setting 
# `bash setup_docker.sh`

apt-get update

apt install -y git
apt-get install -y vim

pip install jupyter

# Port exposing 
apt install -y ufw

ufw allow 8888/tcp

jupyter notebook --generate-config

# USER_ID = "/root

chown -R root:root /root/.jupyter && chmod -R 755 /root/.jupyter

# USER $USER_ID
echo "c.NotebookApp.ip = '0.0.0.0'" | tee -a /root/.jupyter/jupyter_notebook_config.py 
# echo "c.NotebookApp.notebook_dir = '/root/'" | tee -a /root/.jupyter/jupyter_notebook_config.py 
echo "c.NotebookApp.token = u\"\"" | tee -a /root/.jupyter/jupyter_notebook_config.py 
echo "c.NotebookApp.password = u''" | tee -a /root/.jupyter/jupyter_notebook_config.py
echo "c.NotebookApp.iopub_data_rate_limit = 10000000" | tee -a /root/.jupyter/jupyter_notebook_config.py
echo "c.NotebookApp.open_browser = False" | tee -a /root/.jupyter/jupyter_notebook_config.py

echo "alias jupyter_notebook ='jupyter notebook --allow-root'" >> /root/.bashrc

# Python Packages
# OpenCV dependency       
apt-get install -y libsm6 libxext6 libxrender-dev

apt-get install -y git
apt-get install -y vim

pip install opencv-python

pip install -U tensorflow==2.0

pip install pillow 
pip install matplotlib
pip install seaborn
pip install pandas
pip install numpy 
