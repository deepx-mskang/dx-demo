sudo mkdir -p /etc/.dx-demos
sudo sh -c "printf '%s\n' 'DX_MODEL_DECRYPT_PASSWORD=dxDemoDeepx@2617' > /etc/.dx-demos/.secret.env"
sudo chown deepx:deepx /etc/.dx-demos/.secret.env
sudo chmod 600 /etc/.dx-demos/.secret.env


```
./enc -a chacha20-poly1305 -u deepx -p dxDemoDeepx@2617
```
