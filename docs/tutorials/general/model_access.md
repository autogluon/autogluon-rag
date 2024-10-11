# Accessing models through different services in `AutoGluonRAG`. 

Depending on what service you are using to access certain models for different modules, you may need to provide access keys for each service.
1. Using GPT models: 
    1. Create an OpenAI account here: https://openai.com/
    2. Access the API section after logging in. Go to the “API” tab or use this [link](https://platform.openai.com/signup) to access the API dashboard
    3. Select the appropriate billing plan for your account to access OpenAI models. Complete all necessary financial information and billing steps.
    4. Generate and API key and store it in a `txt` file on your device. When using AutoGluon-RAG, make sure to specify the path to this file by setting the `openai_key_file` argument in the config file or through code (Refer to [Setting Parameters for `AutoGluonRAG` through code](https://github.com/autogluon/autogluon-rag/tree/main/documentation/tutorials/general/setting_parameters.md) for more info). 


2. Using AWS Bedrock models: You can either use the [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html) or manually set the AWS Keys in your command line configuration file (`bash_profile` or `zshrc` file). If you are doing it manually, make sure to set the following parameters: 
    - `AWS_ACCESS_KEY_ID`
    - `AWS_SECRET_ACCESS_KEY`
    - `AWS_DEFAULT_REGION`


3. Using Huggingface models: You can use the [Hugging Face Command Line Interface (CLI)](https://huggingface.co/docs/huggingface_hub/en/guides/cli#command-line-interface-cli) to access Hugging Face models. Follow these steps to access Hugging Face models:
    1. Installation
            
            pip install -U "huggingface_hub[cli]"
        
        Once installed, check that the CLI is correctly setup:
                
            huggingface-cli --help

    2. Login to your Hugging Face account
        1. First, create a Hugging Face account [here](https://huggingface.co/join).
        2. Then, obtain an access token using [this](https://huggingface.co/settings/tokens) link. You can find more information about User access tokens [here](https://huggingface.co/docs/huggingface_hub/en/guides/cli#command-line-interface-cli).
        3. Once you have your token, run the following command in your terminal:

                huggingface-cli login

            Enter your access token when prompted. You can optionally use the Hugging Face token as a `git` credential if you plan to use `git` locally and contribute to this package.
