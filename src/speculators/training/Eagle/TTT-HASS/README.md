# Train Time Test HASS 
#### This is the training code for a modified version of the HASS method which creates models that are a modified version of the Eagle 1 architecture.  

## To Run:  

By default this uses the Llama 3.1.8B model.  To use other Llama 3 models, you can simply modify the model directory and configuration file.  However, for models with a different chat template or lm_head location, you need to fix the system prompt in the data script and the name of the lm_head in order to load the lm_head weights correctly.  We provide an alternative path for mistral-small. So far, we only allow llama-based speculator architectures, but these can be used with other verifier architectures as long as you match the shape of the kv cache (for vllm deployment).  This corresponds to matching head_dim, num_key_value_heads and num_attention_heads.  


### Data Generation step: 

1. Modify the directory names and arguments in `gen_data.sh`
2. Make sure that the system prompts and chat template demarkation in the desired file (ultrachat.py or sharegpt.py) are correct
3. Run the script: `./gen_data.sh`
4. Run for each of: sharegpt, and ultrachat sft and gen splits.  
Notes:  For llama 3.1.8B this will generate ~4TB of data on your system.  

### Run training 
1. Modify the directory names and arguments in `train.sh`
2. Run `./train.sh`

### Serve the model with vllm:
1. Convert your saved model with: `convert.sh`
2. Serve the model with: ` VLLM_USE_V1=1 vllm serve meta-llama/Llama-3.1-8B-Instruct --seed 42 -tp 1 --speculative-config '{"model": "llama_eagle", "num_speculative_tokens": 3, "method":"eagle3", "draft_tensor_parallel_size":1}'`



### TODO:  
1. Throw an error if you attempt to create a model that will not be supported in vllm - with the wrong configuration of heads etc.
2. 
