# playing_with_fhir
measure shape polymorphism and nesting of FHIR resources across versions

## usage
```bash
gh repo clone bionicles/playing_with_fhir
cd playing_with_fhir

# go to https://synthetichealth.github.io/synthea/
# download the dstu2, stu3, and r4 datasets 
# extract the json into the data/$VERSION folders

conda create -n py39 python=3.9
conda activate py39
pip install rich, numpy
python fhir.py
```
