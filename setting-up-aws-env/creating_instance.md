Launch an instance
Search for *Deep Learning AMI GPU PyTorch 1.9.0*
Look in Community AMIs and choose Amazon's Amazon Linux 2 AMI.

On Configure Instance Details, make sure you choose the right subnet for your
preferred availability zone.  Notably, p2 instances are not available in
us-east-1f, but they are available in 1e.

On Add Storage, choose the gp3 Volume Type. It's the latest generation, and is
higher performance and slightly cheaper.  When you choose your disk size, keep
in mind that it's fairly easy to increase it later. But you can't shrink it
later.

On Configure Security Group, choose Select an existing security group and then
"ml ssh".

Select the instance. Use Actions / Security / Modify IAM role to give the
instance the IAM role ml-ec2-instance.

Choose an unused Elastic IP or allocate one, and Associate it with the instance.

ssh to the instance. Forward ports and ssh keys if you like. For example:

ssh -i ~/.ssh/deansherssh2021-07-17.pem -L8888:localhost:8888 ec2-user@23.23.188.60
