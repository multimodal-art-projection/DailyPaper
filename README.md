<h1 align="center"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/63839e9962badff4326cf360/k4Q7R4XLDMp_1VF4C6GEd.jpeg" width="25"> M-A-P Daily Paper</h1>
<p align="center">
<a href="https://github.com/DenverCoder1/readme-typing-svg"><img src="https://media.giphy.com/media/Rn26lWjqA0uUU/giphy.gif" width="100"></a>
</p>
<hr/>
<h4 align="center">The <a href=https://m-a-p.ai>M-A-P</a> daily paper project curates and reviews a selection of new papers published daily on arXiv, providing insightful commentary on cutting-edge research across various scientific disciplines.</h4>
<br>

[Click to view previous selection](https://m-a-p.ai/DailyPaper/archived_papers.html).

<hr/>

## 🔥 Paper Today: 17/10/2024

<table class="center">

| Paper | Comments |
|:-------------|:-------------|
| OmnixR: Evaluating Omni-modality Language Models on Reasoning across Modalities | The researcher recommends their OmniBench and shares insights gained from three to four months of dedicated evaluation experience. They suggest that insightful evaluations should naturally possess certain characteristics in data organization: 1) Avoid coupling between model capabilities, focusing on issues that can be improved through architecture or data but may be blind spots. 2) Design evaluation sets for genuinely universal problems, including future issues, but not non-universal ones. 3) Cognitive science concepts are crucial, offering many valuable evaluation perspectives. 4) Evaluations should not fear being solved quickly; benchmarks are consumables designed to guide model improvements, fulfilling their historical mission by identifying issues. |
| Proactive Agent: Shifting LLM Agents from Reactive Responses to Active Assistance | The researcher highly endorses this direction, viewing it as an area where academia can continually accumulate value with measurable outputs from investments. The most valuable general insights involve LLMs' basic abilities, proactively seeking information and simple Agent Instruct capabilities. By continuously summarizing and forming general workflow paths to solve problem types, robust synthetic environments can generate numerous workflows with potential practical value for platforms like Coze or Difny. However, the researcher suggests that while generating workflows is valuable, dedicating efforts to reinforcement learning might be unnecessary at this stage. |
| PRefLexOR: Preference-based Recursive Language Modeling for Exploratory Optimization of Reasoning and Agentic Thinking | The researcher finds this paper, written by an MIT PI in materials science, interesting. It presents an O1-like system concept with several insightful elements, such as using ORPO for iteration, dynamic knowledge graphs, and the notion of thinking tokens, which might serve a function similar to tone words in O1. The researcher humorously adds a citation review: "Identified as aspiring for a Nobel Prize in Physics." |
| Revealing the Barriers of Language Agents in Planning | The researcher acknowledges Blocksworld and TravelPlanner as excellent datasets, suggesting the inclusion of TravelPlanner in the OOD bench due to its wide recognition. Key takeaways align with their internal findings: current open-source models tend to follow instruction intent coarsely rather than decomposing and strictly adhering to complex composite instructions, struggling with combinations. This is identified as a key capability requiring attention. The definitions of episodic memory and parametric memory are also noted as interesting. |
| JudgeBench: A Benchmark for Evaluating LLM-based Judges | The researcher views this as a correct direction, leaning more towards Factuality and Complex Reasoning compared to RewardBench. It is seen as an effective supplement for analyzing Reward Modeling effects. |
| DDIL: Improved Diffusion Distillation With Imitation Learning | - |
| MoE-Pruner: Pruning Mixture-of-Experts Large Language Model using the Hints from Its Router | - |
| Open Domain Question Answering with Conflicting Contexts | The researcher identifies this as a valuable Reasoning Benchmark that incorporates Conflict Context. |
| Understanding the Role of LLMs in Multimodal Evaluation Benchmarks | The researcher acknowledges MMMU's limitations, noting that most questions can be answered without vision-related world knowledge. This insight is deemed important for benchmark design, emphasizing the need to strictly consider whether visual input is necessary, whether spatial awareness is required, or if text descriptions suffice. Three levels of benchmarks are proposed: no visual need, visual need without spatial awareness, and essential spatial awareness. |
| HumanEval-V: Evaluating Visual Understanding and Reasoning Abilities of Large Multimodal Models Through Coding Tasks | The researcher recommends integrating this multimodal coding benchmark into internal visual benchmarks. |
| Is Complex Query Answering Really Complex? | The researcher finds the paper's breakdown of partial reasoning queries versus full reasoning queries significant for language model benchmarks. They suggest that applying this decomposition to problems like HotpotQA could yield valuable insights into the proportion of reasoning versus memorized components in long chain-of-thought solutions. |
| WorldCuisines: A Massive-Scale Benchmark for Multilingual and Multicultural Visual Question Answering on Global Cuisines | - |
| A Scalable Communication Protocol for Networks of Large Language Models | The researcher distills the core insight: frequent communications are handled through traditional protocols, infrequent ones through structured data exchange, and rare communications use natural language. They agree with this approach, advocating for systems that utilize LLMs appropriately without assigning unnecessary active roles for the sake of being fancy. |
| Identifying Task Groupings for Multi-Task Learning Using Pointwise V-Usable Information | The researcher sees potential in adapting this as an indicator for controlling SFT data distribution. They suggest exploring methods from multi-task learning task number scaling for potentially useful approaches. |
| Counterfactual Generative Modeling with Variational Causal Inference | - |
| Exploring Model Kinship for Merging Large Language Models | The researcher finds this paper interesting, defining a new direction in model kinship. They suggest broader applications beyond model merging and note areas for improvement, such as considering inter-layer differences and robustness across different architectures. They propose extending the concept to define cross-model kinship based on semantic similarities of head patterns and activation patterns in complex reasoning cases. |
| Rethinking Visual Counterfactual Explanations Through Region Constraint | The researcher marks this for later reading, noting that the Automated Region Extraction step seems non-essential and abrupt. They find other parts of the definition self-consistent based on their limited diffusion knowledge and plan to study it more carefully later. |
</table>


## 🛠️ Papers This Week 

(Expand to View)
<details>
<summary> <b>15/10/2024</b> </summary>

<table class="center">

| Paper | Comments |
|:-------------|:-------------|
| TMGBench: A Systematic Game Benchmark for Evaluating Strategic Reasoning Abilities of LLMs | TMGBENCH includes 144 types of games based on Robinson-Goforth topologies, each containing multiple instances. These games can be further organized into sequential, parallel, and nested forms. Evaluation metrics are designed, effectively reflecting fluid intelligence and dynamic scalability. Similar to previous benchmarks, a decisive gap between open-source models and models like Claude and GPT-4 was observed. Forwarded to @Peng Tao @Yang Guang. Model behavior traits may be critical, as BoN's ability to select good cases depends on sufficient sampling diversity. However, open-source models lack dynamic thinking, possibly making inference scaling harder for these models. Highly recommended for reading, with an emphasis on improving dynamic pattern composition. |
| STACKFEED: Structured Textual Actor-Critic Knowledge Base Editing with FeedBack | A reverse thinking approach, where instead of using Actor-Critic to modify LLM, it is used to modify KB. This makes a lot of sense. Such reverse thinking seems much needed. |
| Embedding Self-Correction as an Inherent Ability in Large Language Models for Enhanced Mathematical Reasoning | Developed by Tencent, this combines OpenCodeInterpreter with self-improvement for math, though it's fairly average. Many methods introducing strong verifiers to support CoT already exist. |
| AFlow: Automating Agentic Workflow Generation | Personally, I'm not fond of using an agent workflow to solve all problems, but this paper feels like an exception with fundamental value. The abstraction suggests that if a specific workspace definer exists, it could sequentially sample and generate workspace descriptions. This has high theoretical value for frameworks like Coze and Difny. Recommended for follow-up. |
| VideoAgent: Self-Improving Video Generation | This tackles video generation with self-improvement based on external feedback, though it was only tested in robotic scenarios using MetaWorld, iTHOR, and BridgeData V2 datasets. Limited knowledge in this field makes it hard to confirm whether the title might be an overclaim, as "AIGC" comes to mind. |
| OpenR: An Open Source Framework for Advanced Reasoning with Large Language Models | A framework with extensive support according to the paper, including PRM support for scoring and strategy selection. Key formalizations are generalizable, with few unconventional techniques. Its practical usability may depend on ease of use. Worth exploring. |
| Alignment Between the Decision-Making Logic of LLMs and Human Cognition: A Case Study on Legal LLMs | An evaluation framework for step-wise CoT generation quality, which defines irrelevant, incorrect, and correct tokens. It could extend to a framework for evaluating BoN/MCTS/CoT performance by measuring step repetition, CoT reliability, and search space. The interaction framework feels unnecessary. |
| Mechanistic Interpretability for AI Safety A Review | A primer on mechanical interpretability, offering a historical overview. It could be a worthwhile read for those interested in the basics. |
| Zero-shot Commonsense Reasoning over Machine Imagination | Involves generating text QA pairs from a knowledge base, followed by text-to-image generation for a VQA dataset. An intern's project that aligned CLIP embeddings but seemed scooped. Though it improves commonsense reasoning, the work isn't particularly solid. Issues with SFT and pretraining data alignment are worth further exploration. |
| Two Heads Are Better Than One: A Multi-Agent System Has the Potential to Improve Scientific Idea Generation | Reads like a science fiction story, but it's quite fun. The idea of generating a digital researcher and simulating collaborative research sounds interesting. It raises curiosity about personal digital researcher performance. |
| HART: Efficient Visual Generation with Hybrid Autoregressive Transformer | MIT's autoregressive diffusion-based image generation model. Strongly recommended for a detailed read. |
| LVD-2M: A Long-take Video Dataset with Temporally Dense Captions | Congrats! This synthetic long video generation data is highly valuable. |
| TemporalBench: Benchmarking Fine-grained Temporal Understanding for Multimodal Video Models | A valuable benchmark for video understanding. However, the videos might be too short, with the longest ones under 10 minutes. The study found that performance saturates at 8-16 frames, which is a key takeaway. Recommended reading. |
| On Information-Theoretic Measures of Predictive Uncertainty | Haven't had the chance to study the formulas thoroughly. Marked for later review. Transformation between Equation 1 and 5 seems fine. These measures might be critical for improving preference data efficiency. |
| When Attention Sink Emerges in Language Models: An Empirical View | Highly recommended reading. Although the sample size isn't large, the experiments are rigorous. Key takeaways: (1) weight decay encourages attention sinks, (2) larger training datasets exacerbate attention sinks, (3) random and repeated sequences influence attention sinks. This supports sentence-level deduplication and the removal of random sequences. |
| Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models | Simple and effective method using residuals and decoupling high-resolution adaptation to solve reconstruction accuracy issues. The approach feels like broadening the information bottleneck. Highly recommended for reading. |
| SeedLM: Compressing LLM Weights into Seeds of Pseudo-Random Generators | Training-free model compression method without calibration data. Introduces bypass parameters to reconstruct weight blocks, aligned with MoE fine-grained compression thoughts. |
| Adapt-∞: Scalable Lifelong Multimodal Instruction Tuning via Dynamic Data Selection | The idea of dynamically selecting and pooling new SFT data is a good system demo concept, although it feels underdeveloped. Recommended reading. |
| BookWorm: A Dataset for Character Description and Analysis | Potentially important for role-playing, containing character descriptions and in-depth analysis with high confidence. |
| Evaluating Semantic Variation in Text-to-Image Synthesis: A Causal Perspective | An interesting new benchmark that measures the causal effects of semantic variation in text-to-image models. The key takeaway: cross-modal alignment in UNet or Transformers is critical, and text encoder capability isn't the sole determining factor. |
| Predicting from Strings: Language Model Embeddings for Bayesian Optimization | GDM's exploratory work using language model embeddings for Bayesian optimization. Larger models perform better, showing higher capacity and better inductive bias. |
| LOBG: Less Overfitting for Better Generalization in Vision-Language Model | Through fine-grained filtering and hierarchical logic distillation, this study improves vision-language model generalization. Although there is high exploratory value, the approach feels basic. |
| α-DPO: Adaptive Reward Margin is What Direct Preference Optimization Needs | Marked for formula review. |
| FormalAlign: Automated Alignment Evaluation for Autoformalization | Addressing formalization issues in lean translation. Current metrics lack persuasiveness. Semantic analysis in this context is questionable. |
| Scalable Multi-Domain Adaptation of Language Models using Modular Experts | An early 2023 project on modular expert systems with a very rough implementation. It aligns with the idea of training multiple experts and using a routing system for unseen tasks. The approach has potential for edge-side research. |
| Collu-Bench: A Benchmark for Predicting Language Model Hallucinations in Code | A valuable benchmark addressing hallucination in code generation. The key takeaway: keywords, identifiers, and type identifiers are most prone to hallucination. |
| Retrieval Instead of Fine-tuning: A Retrieval-based Parameter Ensemble for Zero-shot Learning | Similar to another paper, it offers a retrieval-based parameter ensemble for zero-shot learning. |
| Gradient-Free Neural Network Training on the Edge | A gradient-free training method. A quick glance shows that gradient replacement involves an intermediate tensor. |
| MoIN: Mixture of Introvert Experts to Upcycle an LLM | Similar to a previous paper, exploring introvert experts for LLM. |
| Can In-context Learning Really Generalize to Out-of-distribution Tasks? | Solid experiments but similar to a previous paper, showing ICL mainly retrieves implicit functions from pretraining. |
| Reconstructive Visual Instruction Tuning | By introducing reconstructive visual instruction tuning, this significantly improves fine-grained understanding in LMMs and reduces hallucination. It aims to balance low-level visual information, though it feels somewhat esoteric. |
| Boosting Deductive Reasoning with Step Signals In RLHF | - |
| MIRAGE: Evaluating and Explaining Inductive Reasoning Process in Language Models | Potentially a good benchmark for evaluating fluid intelligence. |
| Fine-grained Attention I/O Complexity: Comprehensive Analysis for Backward Passes | A detailed analysis of attention mechanism I/O complexity. Worth revisiting. |

</table>

</details>

<hr/>

<details>
<summary> <b>14/10/2024</b> </summary>

<table class="center">

| Paper | Comments |
|:-------------|:-------------|
| Editing Massive Concepts in Text-to-Image Diffusion Models | This work is about concept editing on scalable image. But the results show the model lacks robustness. The collection of 1,000 potentially problematic concepts presents a meaningful problem space. But it remains unclear how this model applies to real-world applications. In order to prevent generating outdated, copyrighted, incorrect, and biased content, it's crutial to know these errors during the generation. Diffusion-based models demonstrate limited concept-based world knowledge, leading to an unsustainable pattern of continuous patching. While this approach could potentially be applied to prevent copyrighted image generation. But copyrighted images should not be used in the training process. Additionally, the study lacks experimental validation for potential model collapse issues, suggesting room for methodological improvement. The ICEB benchmark for evaluating concept-based Image Editing represents a significant contribution, offering unprecedented scale in this domain. |
| Promptly Yours? A Human Subject Study on Prompt Inference in AI-Generated Art | Figures 11-14 reveal significant insights: Diffusion models demonstrate limited generalization from original prompts to generated images, with both humans and AI showing inability to recall original prompts. |
| KV PREDICTION FOR IMPROVED TIME TO FIRST TOKEN | It's one of the interesting works from Apple recently. It proposes a small auxiliary model which is used to process the prompt and produce an approximation of the KV cache used by a base model. The commenter mentions an idea which hasn't been realised. It's to build a small model to predict what experts can be activated by MoE based on the given tokens. But it's time-costly in loading models and putting them back. A potential benefit of this approach is enabling the utilization of models with larger total parameters while operating under memory constraints. The commenter guesses KVP-C and KVP-LP here suggest robust activation patterns across different model sizes. The models can retain similar activation patterns with different model size and different pruning. |
| UNIQ: Offline Inverse Q-learning for Avoiding Undesirable Demonstrations | Well-structured research addressing sparse expert data in offline imitation learning. Proposes maximizing distance between readily available undesired demonstrations rather than minimizing distance to expert data. Potential applications for RLHF merit further investigation. |
| Can Looped Transformers Learn to Implement Multi-step Gradient Descent for In-context Learning? | Robust Mechanical Interpretability study using synthetic data. Demonstrates single-layer Transformer capability for multi-step algorithms, extending to multi-layer implementations. Shows OOD algorithmic generalization, particularly in covariance matrix testing. Implications for pre-training algorithm circuits and potential for high-level extrapolation through proper head activation. |
| Koala-36M | Potentially valuable video dataset with significant resource investment. Further investigation recommended. |
| Baichuan-Omni Technical Report | Focuses on pre-training developments. Visual Encoder-Projector implementation warrants investigation for semantic token conversion efficiency. Stage 2 synthetic QA and OCR quality improvements align with traditional CLIP retraining approaches. Audio processing methodology based on image-like dependency learning raises concerns. |
| SimpleStrat: Diversifying Language Model Generation with Stratification | Quantifies model response diversity through ConvergeQA averaging 28 answers. Potential benchmark for evaluating model tendencies between deep search patterns and exploration. |
| Agents Thinking Fast and Slow: A Talker-Reasoner Architecture | Google's agent framework demonstrates talker-reasoner verification patterns. Experimental observations across HotpotQA/Collie/AIME/Usaco/LiveCodeBench suggest benchmark-specific thought patterns, potentially indicating synthetic data influence rather than learned behavior. |
| ∀uto∃∨∧L | Candidate for scalable fluid intelligence benchmark. |
| NoVo: Norm Voting off Hallucinations with Attention Heads in Large Language Models | Investigates inference-time head control for behavior patterns. MCQ application appears limited. |
| The structure of the token space for large language models | Proposes token subspace as stratified manifold rather than manifold. Experimental validation requires further investigation. |
| Towards Cross-Lingual LLM Evaluation for European Languages | European minor language benchmark collection. |
| CryoFM | Acknowledgment of achievement. |
| VERIFIED | Challenging precise video frame extraction benchmark, enhancing coarse-grained image descriptions with fine-grained detail correspondence. |
| PoisonBench | Collaborative work between Anthropic and Renmin University identifying: 1) Parameter scaling's limited impact on poisoning resistance, 2) Log-linear relationship between attack effects and poison ratio, 3) Data poisoning generalization to extrapolated triggers. |
| On the token distance modeling ability of higher RoPE attention dimension | Valuable analysis of RoPE dimension contributions to attention heads. Figure 9 demonstrates significant top 10% head contribution. Implications for head activation sparsity in long-text scenarios require further investigation. |
| ZipVL | Implements dynamic token proportion determination through attention scores, utilizing important tokens in pre-filling stage. Potential applications for Long Video Understanding given redundant information characteristics. |
| Scaling Laws for Predicting Downstream Performance in LLMs | Solution lacks optimization for downstream performance fitting issues. Formula 5.1 warrants investigation. Suggests potential for refined data mixture law development. Alternative regression approaches using MLP or advanced models may improve inference efficiency. References "Does your data spark joy?" regarding performance improvements through domain upsampling in final training phases. |


</table>

</details>

<hr/>

If you are intereted in the work published by us, please navigate to our [full paper list](https://huggingface.co/collections/m-a-p/m-a-p-full-paper-list-65e070a694c7b01c5547fbff).
