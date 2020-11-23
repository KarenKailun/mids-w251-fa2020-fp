#!/bin/bash

# make sure you have the w251-fp-user.keys file from Justin, in the same directory
# where you're executing this script. once this is run successfully, you should be
# able to pass the `--profile w251-fp-user` flag to aws s3 commands so that you
# can access our bucket. for example, this command will download all the files in
# our bucket to the current directory:
#
#     aws s3 sync s3://mids-w251-fp/data . --profile w251-fp-user

if [ -f ./w251-fp-user.keys ]; then
    ACCESS_KEY=$(jq -r '.AccessKey.AccessKeyId' w251-fp-user.keys)
    SECRET_KEY=$(jq -r '.AccessKey.SecretAccessKey' w251-fp-user.keys)
    aws configure set aws_access_key_id $ACCESS_KEY --profile w251-fp-user
    aws configure set aws_secret_access_key $SECRET_KEY --profile w251-fp-user
    aws configure set region us-east-2 --profile w251-fp-user
else
    echo "Cannot find the w251-fp-user.keys file in this directory."
fi
