<h1 align="center"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/63839e9962badff4326cf360/k4Q7R4XLDMp_1VF4C6GEd.jpeg" width="25"> M-A-P Daily Paper</h1>
<p align="center">
<a href="https://github.com/DenverCoder1/readme-typing-svg"><img src="https://media.giphy.com/media/Rn26lWjqA0uUU/giphy.gif" width="100"></a>
</p>
<hr/>
<h4 align="center">The <a href=https://m-a-p.ai>M-A-P</a> daily paper project curates and reviews a selection of new papers published daily on arXiv, providing insightful commentary on cutting-edge research across various scientific disciplines.</h4>
<br>

[Click to view previous selection](https://m-a-p.ai/DailyPaper/archived_papers.html).

<hr/>

## 🔥 Paper Today: 15/10/2024

<table class="center">

| Paper | Affiliation | Comments |
|:-------------|:-------------|:-------------|
| **TMGBench: A Systematic Game Benchmark for Evaluating Strategic Reasoning Abilities of LLMs** | HKU, Harbin Institute of Technology | TMGBench includes 144 types of games based on the Robinson-Goforth topology, with each type containing multiple instances. These games can be further organized into sequential, parallel, and nested complex forms. Evaluation metrics designed for these games reflect dynamic and scalable assessments of fluid intelligence, highlighting significant gaps between open-source models and models like Claude and 4o. It is recommended to emphasize the importance of dynamic pattern composition capabilities. |
| **STACKFEED: Structured Textual Actor-Critic Knowledge Base Editing with Feedback** | Microsoft Research | This presents an interesting reversal of thinking, as it involves modifying Knowledge Bases with Actor-Critic approaches, which sounds quite reasonable. There appears to be a strong need for such a reverse thought process. |
| **Embedding Self-Correction as an Inherent Ability in Large Language Models for Enhanced Mathematical Reasoning** | Tencent | This work from Tencent employs OpenCodeInterpreter and Self-Improvement for mathematical applications. While it may seem unremarkable, many methods currently incorporate strong verifiers to support CoT (Chain of Thought). |
| **AFlow: Automating Agentic Workflow Generation** | DeepWisdom, The Hong Kong University of Science and Technology (Guangzhou) | The author expresses a personal disinterest in the direction of a single agent workflow solving all problems; however, this paper feels like an exception and is quite fundamental. The theoretical value of this paper for frameworks like Coze and Difny is high. It raises an interesting abstraction: if a dedicated Workspace definer could generate numerous sequential workspace descriptions from similar inputs and outputs, could it initialize potential workflows effectively? Recommended for reading and following up. |
| **VideoAgent: Self-Improving Video Generation** | University of Waterloo, IIT Kharagpur, Google DeepMind | This work focuses on self-improving video generation based on external feedback. While it appears to be somewhat effective, it has only been tested in robotic scenarios using datasets such as MetaWorld, iTHOR, and BridgeData V2. The author's familiarity with the field is limited, making it difficult to determine if the title may be somewhat overstated, as it inevitably evokes thoughts of AIGC. |
| **OpenR: An Open Source Framework for Advanced Reasoning with Large Language Models** | University College London, University of Liverpool, Shanghai Jiao Tong University | This work is framework-oriented. The author hasn't yet reviewed the repository but has read the paper introduction, which suggests comprehensive and solid support. The PRM accommodates settings for both final score selection and overall scores, while the strategy options support majority vote and maximum scoring. The formalization of the problem presented is quite generalizable. Overall, it's well summarized, although details on construction and testing at this early stage seem less meaningful. |
| **Alignment Between the Decision-Making Logic of LLMs and Human Cognition: A Case Study on Legal LLMs** | University of Geneva, Google DeepMind, École Normale Supérieure | This appears to be a framework that could extrapolate to evaluate the quality of conventional multi-step CoT generation, defining unrelated tokens, erroneous tokens, and correct tokens. It may be beneficial to refine the granularity of analysis based on LLM-as-a-judge principles. The framework concerning interaction seems superfluous, with some steps and standards compared to standard CoT extending into specialized evaluation frameworks. |
| **Mechanistic Interpretability for AI Safety A Review** | University of Amsterdam | The paper offers a basic overview of mechanical interpretability concepts and history, which may be of interest to some readers. |
| **Zero-shot Commonsense Reasoning over Machine Imagination** | - | This study generates text QA pairs from a knowledge base and uses text-to-image models to create corresponding images, forming a VQA dataset that includes text, answer options, and images. The work, conducted by an intern, may suggest a sense of being scooped. The intern employed a dual-tower model that aligns text with CLIP embeddings, ultimately creating synthetic image descriptions to match image embeddings. While the idea may seem inexpensive, it does demonstrate an enhancement in commonsense reasoning ability, which is significant. |
| **Two Heads Are Better Than One: A Multi-Agent System Has the Potential to Improve Scientific Idea Generation** | Shanghai Artificial Intelligence Laboratory | This concept, resembling a sci-fi novel, is quite enjoyable. It involves generating a digital representation for each researcher and simulating how different researchers might collaborate on research. There is curiosity about how the digital representation of one’s own researcher would appear and how it might analyze the contributions of specific agents. |
| **HART: Efficient Visual Generation with Hybrid Autoregressive Transformer** | MIT, NVIDIA, Tsinghua University | This autoregressive diffusion-based image generation model from MIT is highly recommended for thorough reading. The author hasn't had sufficient time yet and intends to mark it for future review. |
| **LVD-2M: A Long-take Video Dataset with Temporally Dense Captions** | The University of Hong Kong, ByteDance | Congratulations on this valuable synthetic long video generation dataset! |
| **TemporalBench: Benchmarking Fine-grained Temporal Understanding for Multimodal Video Models** | University of Wisconsin-Madison, Microsoft Research | This benchmark is highly valuable for video understanding. The only concern may be its brevity, as it appears relatively short, with long videos defined as under 10 minutes. A significant takeaway from the research is that model performance tends to saturate between 8-16 frames. It is recommended for reading. |
| **Introducing an Improved Information-Theoretic Measures of Predictive Uncertainty** | Johannes Kepler University Linz | The author has not yet had time to thoroughly examine the formulas but intends to mark this work for future reference. A preliminary check of the transformations between Equation 1 and Equation 5 did not reveal any immediate issues. It is believed that this type of metric could play a crucial role in enhancing the data efficiency of preference data, although no solid ideas have emerged yet. |
| **When Attention Sink Emerges in Language Models: An Empirical View** | Sea AI Lab, National University of Singapore | This paper is highly recommended for reading. Aside from a possible size limitation, the experiments are extensive and rigorous. The author plans to conduct a detailed study tomorrow but notes several key takeaways: (1) weight decay encourages attention sinks; (2) the larger the training data, the more pronounced the model's attention sink behavior; (3) random and repeated sequences significantly impact the emergence of attention sinks. |
| **Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models** | MIT, Tsinghua University, NVIDIA | This approach is simple and effective, utilizing residual connections and introducing decoupled high-resolution adaptations to address reconstruction accuracy issues. The work appears to widen the information bottleneck. After a long time, it has been a refreshing experience for the author, who has a background in NLP, to read a CV-oriented solution in detail. It is highly recommended for reading. |
| **SeedLM: Compressing LLM Weights into Seeds of Pseudo-Random Generators** | Apple, Meta | This training-free model compression method does not utilize calibration data, focusing instead on weight blocks and incorporating bypass parameters for reconstruction. This approach seems to align well with the idea of compressing fine-grained MoE (Mixture of Experts) strategies. The author intends to review it in detail tomorrow. |
| **Adapt-∞: Scalable Lifelong Multimodal Instruction Tuning via Dynamic Data Selection** | UNC Chapel Hill | This data selection method can also be applied to text SFT (Supervised Fine-Tuning). Essentially, it proposes a pool of SFT data to measure distribution, with new data being added continuously and quality selections made based on distribution. This concept appears to be a promising system-level demo, although the execution seems rather cursory. It is recommended for reading. |
| **BookWorm: A Dataset for Character Description and Analysis** | University of Edinburgh | This dataset may be quite important for role-playing, as it contains several reliable character descriptions and in-depth analyses. |
| **Evaluating Semantic Variation in Text-to-Image Synthesis: A Causal Perspective** | Fudan University, Hong Kong University of Science and Technology (Guangzhou) | This new benchmark for text-to-image synthesis measures the causal impact of semantic variations on models. The design seems clever, with a significant takeaway that cross-modal alignment plays a key role in UNet or Transformers, indicating that the capabilities of text encoders are not the sole determining factor. |
| **Predicting from Strings: Language Model Embeddings for Bayesian Optimization** | UCLA, Google DeepMind, Google | This work from GDM explores the nature of the direction, which seems to stem from the encouragement of such endeavors. It involves embedding experimental inputs as feature vectors using language models and applying them in context regression models. By pre-training a transformer-based regression model on extensive offline evaluation data, it achieves uncertainty-aware numerical predictions for new objective functions. |
| **LOBG: Less Overfitting for Better Generalization in Vision-Language Model** | Xi’an Jiaotong University | This research significantly improves the generalization capabilities of vision-language models by filtering out irrelevant fine-grained information and maintaining structural topology and hierarchical logic during distillation. There has been a recent increase in studies of this nature. If MLLM represents human visual perception of the external world, it is essential to sift through redundant information purposefully. While the methods in this article seem rather basic, it lacks depth. |
| **α-DPO: Adaptive Reward Margin is What Direct Preference Optimization Needs** | University of Science and Technology of China | The author has not yet had the opportunity to derive formulas, but intends to mark this for future formula practice. ｜
| **FormalAlign: Automated Alignment Evaluation for Autoformalization** | HKU, Cambridge, Huawei Noah's Ark Lab | The issues addressed in this paper are significant. The challenges faced in Lean arise primarily from inadequate formalization. The metrics used are not sufficiently persuasive; the cross-entropy approach appears somewhat arbitrary. The difficulties in Lean translation stem from grammatical details, and semantic analysis could reveal the oddity of certain outcomes. |
| **Scalable Multi-Domain Adaptation of Language Models using Modular Experts** | UC Berkeley, Google | A recommendation for an early work called Deep-ICL, which has room for improvement. Although the initial approach was not very strong, the fundamental ideas align closely with this direction. It might be among the earliest papers in this area. The proposed method involves using a backbone to integrate several trained expert modules that can be activated based on the input information, along with training a routing system for unseen tasks. This direction may be worth researching for edge applications. |
| **Collu-Bench: A Benchmark for Predicting Language Model Hallucinations in Code** | Purdue University | This benchmark addresses a valuable problem and format. An important takeaway is that "keywords, identifiers, and type identifiers are the most prone to hallucinations." |
| **Retrieval Instead of Fine-tuning: A Retrieval-based Parameter Ensemble for Zero-shot Learning** | Massachusetts General Hospital, Harvard Medical School | This is another paper similar to the previous one. |
| **Gradient-Free Neural Network Training on the Edge** | Bosch Centre for Artificial Intelligence | The gradient-free training approach involves replacing the gradient with an intermediate tensor, deciding to flip related nodes. Further detailed study is planned for tomorrow. |
| **MoIN: Mixture of Introvert Experts to Upcycle an LLM** | - | This paper is also similar to the previous one. |
| **Can In-context Learning Really Generalize to Out-of-distribution Tasks?** | Peking University, MIT | The experiments are quite solid, but the conclusions are similar to those in Jiaoda Li's paper, which also utilized GPT-2. Beyond the unnecessary mathematical details, the main points are: 1. ICL is fundamentally about retrieving the most relevant implicit functions (or patterns) learned during pretraining to solve problems. 2. Learning new input-output mappings is quite challenging. This paper does not extend to cases with multiple pattern combinations or analyze the data intensity required for pattern learning, which is a significant challenge in this direction. |
| **Reconstructive Visual Instruction Tuning** | Chinese Academy of Sciences | By introducing reconstructive visual instruction tuning, the fine-grained understanding of LMMs has been significantly enhanced, and hallucination phenomena have been reduced. The focus seems to be on avoiding the introduction of excessive low-level visual information. Balancing this aspect appears to be a delicate matter, bordering on the philosophical. This can be viewed alongside the earlier paper. |
| **Boosting Deductive Reasoning with Step Signals in RLHF** | Baichuan AI | This paper has not yet been reviewed but is noted for future reference. |
| **MIRAGE: Evaluating and Explaining Inductive Reasoning Process in Language Models** | Chinese Academy of Sciences | This paper is forwarded to Yang Guang and Peng Tao for consideration. It serves as a potential benchmark for evaluating fluid capabilities. The key takeaway aligns with the previous paper, highlighting that LLMs essentially learn a set of patterns and perform pattern retrieval and scheduling during inference. |
| **Fine-grained Attention I/O Complexity: Comprehensive Analysis for Backward Passes** | Stevens Institute of Technology, University of Wisconsin-Madison, UC Berkeley, University of Pennsylvania | This paper analyzes the I/O complexity of attention mechanisms, indicating it is worth studying further. Noted for future reference. |
| **SeRA: Self-Reviewing and Alignment of Large Language Models using Implicit Reward Margins** | KAIST AI, Amazon | - |
| **Inference and Verbalization Functions During In-Context Learning** | Stanford University | - |
| **Nudging: Inference-time Alignment via Model Collaboration** | University of California Irvine | - |
| **Enhancing Multi-Step Reasoning Abilities of Language Models through Direct Q-Function Optimization** | ByteDance | - |
| **One Step at a Time: Combining LLMs and Static Analysis to Generate Next-Step Hints for Programming Tasks** | JetBrains Research | - |
| **The Same But Different: Structural Similarities and Differences in Multilingual Language Modeling** | Brown University | - |
| **Automated Rewards via LLM-Generated Progress Functions** | Stanford University | - |
| **ACER: Automatic Language Model Context Extension via Retrieval** | CMU, UIUC | - |
| **REDO: Execution-Free Runtime Error Detection for Coding Agents** | University of Pennsylvania | - |
| **Focus on Your Question! Interpreting and Mitigating Toxic CoT Problems in Commonsense Reasoning** | University of Chinese Academy of Sciences | - |
| **DARE the Extreme: Revisiting Delta-Parameter Pruning For Fine-Tuned Models** | University of British Columbia | - |




</table>


## 🛠️ Papers This Week 

(Expand to View)

<details>
<summary> <b>14/10/2024</b> </summary>

<table class="center">

| Paper | Affiliation | Comments |
|:-------------|:-------------|:-------------|
| Editing Massive Concepts in Text-to-Image Diffusion Models | HKU, THU | Editing Massive Concepts in Text-to-Image Diffusion Models addresses the problem of scalable batch image editing based on concepts. However, the approach does not appear to be particularly robust. The authors have collected 1,000 potentially problematic concepts, which are meaningful, but there is uncertainty regarding how this method is intended to be applied in practical scenarios. To effectively avoid issues such as copyright infringement, bias, or factual inaccuracies in generated outputs, the model first needs to recognize when an error has occurred before it can take steps to correct it. This is the critical challenge. Diffusion-based models do not seem to possess a strong concept-based understanding of the world. The current solution appears to rely on continuously patching problems, which is not a sustainable long-term strategy. However, the approach could be useful in preventing the generation of copyrighted images. That being said, ideally, models should not be trained on copyrighted images in the first place. Additionally, the paper does not provide experimental validation for the issue of model collapse. There is room for the method to be more rigorous. The ICEB (Image Concept Editing Benchmark) used to evaluate concept-based image editing is a valuable contribution—such a large-scale benchmark has not been seen before. |
| Promptly Yours? A Human Subject Study on Prompt Inference in AI-Generated Art | University of Oklahoma, University of Texas | Figures 11-14 are particularly interesting. They highlight that diffusion models haven't generalized well from the original prompts to the generated images. In fact, neither humans nor AI are able to recall the original prompts accurately. |
| KV Prediction For Improved Time to First Token | Apple | Apple has recently released several intriguing works. One of them uses a small model to predict approximate values for the KV-cache of a larger model. There was a previous proposal, which wasn’t implemented, that suggested using a small model to predict which experts in a Mixture of Experts (MoE) model might activate based on the input tokens. However, it was reconsidered due to the time cost of loading and unloading experts. The advantage of this approach would be the potential to use a larger MoE model with limited GPU memory. Furthermore, it is unclear whether the KVP-C and KVP-LP methods suggest that models of different sizes, trained on the same data, learn robust activation patterns. Even after pruning or when working with models of varying sizes, the activation patterns seem to remain largely consistent. |
| UNIQ: Offline Inverse Q-learning for Avoiding Undesirable Demonstrations | Singapore Management University | The writing is solid, and the motivation is clear. The issue of sparse expert data in offline imitation learning is real. Instead of minimizing the distance to expert data, the paper proposes maximizing the distance between undesirable demonstrations, which seems like a straightforward idea, particularly for RLHF (Reinforcement Learning from Human Feedback). It’s unclear if this has been explored before; a more thorough survey could be done in the future, along with revisiting the formulas. It's worth marking for further investigation. |
| Can Looped Transformers Learn to Implement Multi-step Gradient Descent for In-context Learning? | Google Research | This work is a well-executed study on mechanical interpretability using synthetic data. However, it suffers from a common issue in such research—the lack of in-depth data analysis, as most of the conclusions are qualitative rather than quantitative. The qualitative finding is that a single-layer Transformer can express multi-step algorithms, which is extended to show that multi-layer Transformers can learn multi-step algorithms. While this is a solid contribution, it doesn't provide a deep understanding of how multi-step algorithms are reflected in the model's parameters. Additionally, the paper demonstrates generalization to out-of-distribution (OOD) algorithms. The model was trained with data using a unit covariance matrix but tested with different covariance matrices, and the looped transformers still managed to achieve low loss, indicating strong generalization across distributions. This raises a hypothesis that pre-training may also involve learning many such algorithmic loops, where multi-hop reasoning could be seen as a form of algorithm. If these parts of the model parameters could be activated—such as in cases where data encoded as code has learned a divide-and-conquer method—there’s potential for such reasoning to generalize to broader data. However, this would require the correct activation of attention heads without overfitting to input patterns. This is recommended reading. |
| Koala-36M: A Large-scale Video Dataset Improving Consistency between Fine-grained Conditions and Video Content | Kuaishou Technology, Shenzhen University, Tsinghua University | This appears to be a highly valuable video dataset, and it seems a significant amount of money was invested in it. Further research is recommended. |
| Baichuan-Omni Technical Report | Baichuan Inc, Westlake University | Baichuan continues to justify its work in pre-training, which appears somewhat basic. The text performance is underwhelming. However, there could be some valuable insights regarding how the encoder is handled. The use of a projector following the visual encoder might improve the efficiency and effectiveness of converting visual input into semantic tokens, and it seems reasonable overall. Stage 2 involves adding synthetic QA and high-quality OCR, which is in line with how others retrain CLIP. If this approach could replace the need to retrain CLIP and caption models, it would be quite valuable. The audio method, which seems to be based on the early understanding of audio being processed similarly to images, is less convincing. It’s disappointing that the paper didn’t test on OmniBench. |
| SimpleStrat: Diversifying Language Model Generation with Stratification | UC Berkeley | This work provides a quantitative assessment of whether models can respond diversely from multiple perspectives. They designed a benchmark, ConvergeQA, which shows an average of 28 possible answers. This could help verify if a model tends to favor depth-first and fixed patterns or if it is more inclined toward exploration. It seems like an interesting observation benchmark and is recommended for further reading. |
| Agents Thinking Fast and Slow: A Talker-Reasoner Architecture | Google DeepMind | Google’s agent framework seems unremarkable, with no groundbreaking metrics. However, the behavioral pattern of a “talker” directing a “reasoner” to verify each intermediate step has been evident in earlier work, such as HotpotQA, Collie, AIME, Usaco, and LiveCodeBench. Collie is an outlier, where, in 50 sampled examples, no pre-planned thought was observed, while the other four benchmarks displayed consistent patterns of divide-and-conquer (COT) or retrieving potential pre-existing solutions (UKM). The intriguing part of this experiment is the high degree of convergence in thought patterns within each benchmark. The question remains whether the model learned this behavior or if it simply reflects rigid synthetic data. The latter seems more likely. For instance, in USACO and LiveCodeBench, divide-and-conquer appeared far more frequently than retrieval-based thinking (UKM), even though template-based approaches are common for humans. Additionally, the consistency of thought patterns within each benchmark supports this hypothesis. Returning to this paper, the talker-reasoner approach could potentially represent a fixed pattern in generating similar data in models like o1. |
| ∀uto∃∨∧L: Autonomous Evaluation of LLMs for Truth Maintenance and Reasoning Tasks | Arizona State University | This could serve as a scalable fluid intelligence benchmark. |
| NoVo: Norm Voting off Hallucinations with Attention Heads in Large Language Models | - | This research explores controlling behavior by manipulating attention heads during inference. It’s an intriguing concept, though the use of this method to address multiple-choice questions (MCQs) feels a bit underwhelming. It’s disappointing that domestic researchers are focusing on leaderboard-hacking; at the very least, MMLU should have been used as a benchmark. |
| The Structure of the Token Space for Large Language Models | American University | The paper argues that token subspace is a stratified manifold rather than a manifold. However, the experiments do not seem particularly robust. It’s unclear what practical significance this conclusion has, so it’s worth marking for future consideration. |
| Towards Cross-Lingual LLM Evaluation for European Languages | TU Dresden, Fraunhofer IAIS | This is a benchmark collection for European minor languages. |
| CryoFM: A Flow-based Foundation Model for Cryo-EM Densities | Bytedance Research | - |
| VERIFIED: A Video Corpus Moment Retrieval Benchmark for Fine-Grained Video Understanding | Tsinghua University | This is a challenging benchmark for fine-grained video moment retrieval, refining coarse-grained image descriptions into those with more precise detail. It increases the level of difficulty in matching descriptions to specific frames. |
| PoisonBench: Assessing Large Language Model Vulnerability to Data Poisoning | Renmin University of China, Anthropic | It’s surprising to see a paper co-authored by Anthropic and Renmin University. The three key takeaways are intriguing: (1) Scaling up parameter size does not inherently enhance resilience against poisoning attacks; (2) There is a log-linear relationship between the effects of the attack and the data poison ratio; (3) Data poisoning effects can generalize to triggers not included in the poisoned data. Recommended reading. |
| On the Token Distance Modeling Ability of Higher RoPE Attention Dimension | Tsinghua University, Tencent Inc | This is a valuable read that analyzes the contribution of various RoPE dimensions to attention heads, identifying positional heads. Figure 9 shows that the top 10% of heads are active, and masking them leads to greater loss than masking the top 5%. This is worth considering in relation to the sparsity of head activation during LLM inference. For long-text cases, it might be necessary to activate more heads. The analysis reveals that the high-dimensional components of most attention heads contribute more to the attention score. The method for extending attention distribution in long texts is worth further exploration. |
| ZipVL: Efficient Large Vision-Language Models with Dynamic Token Sparsification and KV Cache Compression | Zhejiang University, Shanghai AI Laboratory | This paper discusses dynamically determining the proportion of tokens based on attention scores during the pre-filling stage, using only important tokens. It’s seamlessly compatible with existing frameworks. This direction could be highly relevant for long video understanding, as there is often a lot of redundant information. Further research is warranted. |
| Scaling Laws for Predicting Downstream Performance in LLMs | University of Illinois Urbana-Champaign, Amazon | The solution does not directly optimize for the inability to fit downstream performance, and the understanding of downstream datasets' distribution is limited. Section 5.1’s formula is worth investigating. One promising direction is how to segment and fine-tune pre-training data to save on experimental costs. It references a comment: “From an efficiency perspective during inference time, using MLP or a more advanced model for regression would be a better choice; lightgbm + simulation has some flaws.” Fine-tuning learning rate and data schedulers could also be valuable. The paper references a technique where high-quality data is upsampled during the final stages of training (the annealing phase) to boost performance without introducing new data. This approach contrasts with MiniCPM, as it only upsampled existing data but still improved results. Data mixtures might not be fixed ratios either. If unfamiliar with data mixture laws, this paper is worth reading, along with D-CPT Law, RegMix, and a recent paper from Amazon that applies this concept outside of LLMs. |


</table>

</details>

<hr/>

If you are intereted in the work published by us, please navigate to our [full paper list](https://huggingface.co/collections/m-a-p/m-a-p-full-paper-list-65e070a694c7b01c5547fbff).
