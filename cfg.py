def prompt():
    
    f"As a Portrait Artist, evaluate the person in the image:  
    based on the following attributes. 
    Only use the predefined options for each attribute, 
    and provide a concise description in the format shown below.
    Do not add extra traits or indicate inability to evaluate the image. 
    Provide the response exactly as required.
    1. **Gender**: Choose one from [female, male].
    2. **Race**: Choose one from [caucasian, asian, hispanic, 
    african descent, south Asian].
    3. **Age**: Provide the age estimate as 'X years old', 
    where X is between 15 and 80.
    **Output Format Example**:
    Gender: male, Race: caucasian, Age: 20 years old
    Now, based on the observa
    provide the evaluation in the exact format above."
    return prompt
