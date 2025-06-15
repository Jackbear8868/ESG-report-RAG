# ESG-report-RAG
A project in collaboration with the Commerce Development Research Institute to build a Retrieval-Augmented Generation (RAG) system for extracting key terms from ESG reports.

## RAG Prompt
I created the prompt for RAG using this format and asked GPT-o4-mini-high to process it.
```
請你針對指標名稱與關鍵字，產生適合 RAG 的 prompt ，請幫我每個關鍵字產生 5 個 prompt ，請給我 json 的形式

指標名稱："使用再生材料製造之比例中位數與普及度企業數量" 
關鍵字：recycled content ratio%、PCR plastic usage、rPET content、% recycled materials、recycled plastic usage rate、PCR matterial use、post-consumer plastic share、Recycled Content Claim
```
