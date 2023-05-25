# Structure of HTTP Responses 


## (get) /
- for dev 
```json
{
    "message":  "experiment API in: /docs",
    "Configs-CPU": 4,
    "Model ready!": "model-type",
    "API document->": "https://github.com/ewigspace1910/DetectQuestForm4pdf/tree/main/deploy/model"
}
```
1. **message** : is URL to api manipulation page
2. **API document** : URL to API explaination documemt.

## (post) /layout
- overview:
  
```json
{
    "status": "some thing",
    "data": {},
    "excution time" : 1,
    "error": "something"
}

```

1. **status** : is a string and refers to whether the request was successfully operated by the module (`success`, `flase`), 
2. **excution time** : total of time consumption for the request.
3. **data**   : (json) JSON Quiz. it will be empty if status is false
4. **error**  :  show error information if status is false

### **Structure of JSON Quiz**:

Below are the abridged JSONs of the most common quiz types we support. The "..." in the JSON blocks are fields we are ignoring for now as they are not relevant for understanding the structure of the format.

- Open ended question **(OEQ)**
```json
{   "title": "Describe the color of the sky in two pictures below",
    "stimulus": "\\https:pic1.png \n https:pic2.png",
    "prompt"  : "Write your answer",
    "category": "OEQ",
}
```


- Multiple choice question **(MCQ)**
```json
{   "title": "Choose the color of the sky in  picture below",
    "stimulus": "\\https:pic1.png",
    "prompt"  : "Just circle one answer",
    "category": "MCQ",
    "choices" : ["yellow", "red", "blue", "purple"]
}
```  

- Multiple subquestion question **(MSQ)**
```json
{   "title": "Question 10: Look at the picture",
    "stimulus": "\\https:pic1.png",
    "prompt"  : "Answer 3 questions:",
    "category": "MSQ",
    "subquestions" : [
        {"category":"OEQ", "..."},
        {"category":"MCQ", "..."},
        {"category":"OEQ", "..."}
    ]
}
```

### **FULL JSON Quiz Schema**

Below is a full sample of the JSON response that you might receive from our Quiz Import API.

```json
{
  "success": true,
  "filename": "DL 1 Comparison Model.pdf",
  "layout": [
    {
      "name": "heading",
      "title": "Section - 1 ",
      "stimulus": "",
      "prompt": "",
      "category": "SQ",
      "subquestions": [
        {
          "name": "question",
          "title": "Lenny, John and Sam have 820 stickers altogether. Lenny has 55 more stickers than John. John has 4 times as many stickers as Sam. How many stickers does John have?",
          "stimulus": "",
          "prompt": "\n$\\checkmark$ Answer\n\\[\n340\n\\]",
          "category": "OEQ",
          "idx": [
            1,
            208
          ]
        },
        {
          "name": "question",
          "title": "2. Jason spent $\\$ 70$ in 5 days. Each day, he spent $\\$ 3$ more than the day before. How much did he spend on the last day",
          "stimulus": "",
          "prompt": "\n$\\checkmark$ Answer\n\\[\n\\$ 20\n\\]",
          "category": "OEQ",
        },
        {
          "name": "question",
          "title": "5. The figure shows 2 overlapping identical squares (not drawn to scale). $\\frac{2}{5}$ of each square is shaded. What is the ratio of the shaded area to the unshaded area of the figure? Express your answer in its simplest form.",
          "stimulus": "\nhttp://res.cloudinary.com/ddx6w9r4o/image/upload/v1684981517/b3g08obuyuosl7iqn4nw.png",
          "prompt": "\n$\\checkmark$ Answer\n\\[\n2: 6\n\\]",
          "category": "OEQ",
        },
        {
          "name": "question",
          "title": "11. Sinda had a roll of ribbon. She cut the ribbon into shorter pieces of different lengths. When she arranged the shorter pieces o ribbons in ascending order of their lengths, the difference in length between any 2 consecutive pieces was $1.35 \\mathrm{~cm}$.",
          "stimulus": "",
          "prompt": "",
          "category": "SQ",
          "subquestions": [
            {
              "name": "subquestion",
              "title": "(a) The length of the third shortest piece of ribbon was $4.45 \\mathrm{~cm}$. What was the total length of the first 5 pieces of ribbon, starting from the shortest piece? Give your answer in centimetre.",
              "stimulus": "",
              "prompt": "\n$\\checkmark$ Answer\n\\[\n22.25 \\mathrm{~cm}\n\\]",
              "category": "OEQ",
            },
            {
              "name": "subquestion",
              "title": "(b) The difference in length between the shortest piece of ribbon and the longest piece of ribbon was $2.16 \\mathrm{~m}$. How many pieces of ribbons did Sinda cut?",
              "stimulus": "",
              "prompt": "\n$\\checkmark$ Answer\n\\[\n161\n\\]",
              "category": "OEQ",
            }
          ]
        }
      ]
    }
  ],
  "excution time": 69.02719235420227
}

```