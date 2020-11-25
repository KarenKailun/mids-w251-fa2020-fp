#!/bin/bash

echo "Copying contents of s3 bucket locally..."

if [ -d s3 ]; then
    cd s3
else
    mkdir s3
    cd s3
fi

aws s3 sync s3://mids-w251-fp/data . --profile w251-fp-user

echo "Pushing local files to s3..."

aws s3 sync . s3://mids-w251-fp/data --profile w251-fp-user

echo "Sync complete."
cd ..
