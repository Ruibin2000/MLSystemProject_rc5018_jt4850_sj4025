From compute_liqid node CHI@TACC to access the object storage at CHI@TACC



1. Build the session as the lab 1_create_server_nvidia on Training on MLFlow and Ray

   

2. ```
   curl https://rclone.org/install.sh | sudo bash
   
   sudo sed -i '/^#user_allow_other/s/^#//' /etc/fuse.conf
   
   mkdir -p ~/.config/rclone
   nano  ~/.config/rclone/rclone.conf
   
   # run on node-persist
   sudo mkdir -p /mnt/object
   sudo chown -R cc /mnt/object
   sudo chgrp -R cc /mnt/object
   ```
   
3. Add content in **rclone.conf**

   ```
   [kvm_tacc]
   type = swift
   user_id = 84879ad1baddfad9ad8091f7ff467dde56bb2e4c48db8aee2cb5db653ddd38cc
   application_credential_id = a3b806f1e29348eaa0676b22ff4c5308
   application_credential_secret = IxUyjIry-f0I661_My2KjIDGvmMUf5YKAHPR7OeFi-LqfumuB-osEPHY8w7L8eH9Ve52_nlGaIj6JAGudGdyOw
   auth = https://chi.uc.chameleoncloud.org:5000/v3
   region = KVM@TACC
   ```

   
object-persist-group7
4. ```
   cc@node-mltrain-rc5018-nyu-edu:~$ rclone mount chi_tacc:mlops_group7 /mnt/object --read-only --allow-other --daemon
   
   cc@node-mltrain-rc5018-nyu-edu:~$ ls /mnt/object
   Metadatav2.csv  annotation_test.json  annotation_train.json  annotation_val.json  annotations  images
   ```

