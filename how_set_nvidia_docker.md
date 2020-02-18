### Download `nvidia-docker`

참조 레포 

 : https://yahwang.github.io/posts/40

 : https://jybaek.tistory.com/796

- to enable to use docker commands without `sudo`

```bash
sudo usermod -aG docker $USER
```

- ```bash
  curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
    sudo apt-key add -
    
  distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
  
  curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    
  sudo apt-get update
  sudo apt-get install -y nvidia-docker2
  sudo pkill -SIGHUP dockerd
  
  
  ```
