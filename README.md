# How to run the project code:
1. Go to the folder "mocha_code", which is the project's folder.
2. Run the following python scripts, depending on what you want to achieve. A GPU with at least 16GB of memory is required.
   1. _generate_images.py_: will generate the images from the ~3200 captions
   2. _generate.py_: Will generate the captions with Mistral based on the available captions from GPT4
   3. _generate_captions.py_: Will generate captions using BLIP + MOCHa for the new dataset
   4. _evaluate.py_: Will evaluate the hallucination ratio using Mistral on the generated captions from generate_captions.py