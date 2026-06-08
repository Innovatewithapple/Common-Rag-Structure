from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import _faithfulness,_answer_relevance,_context_precision,_context_recall,_answer_correctness
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer,util


class RAGEvaluator:
    def __init__(self,groq_api_key):
        self.groq_llm = LangchainLLMWrapper(ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=groq_api_key
        ))

        self.embeddings = HuggingFaceEmbeddings(
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.st_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.rouge = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'])

    def ragas_eval(self,questions,answers,contexts,ground_truths):
        data = {
            'question':questions,
            'answer':answers,
            'contexts':contexts,
            'ground_truth':ground_truths
        }
        dataset = Dataset.from_dict(data)
        results = evaluate(
            dataset=dataset,
            metrics=[_faithfulness,_answer_relevance,_context_precision,_context_recall,_answer_correctness],
            llm=self.groq_llm,
            embeddings=self.embeddings
        )
        return results
    
    def rouge_eval(self,hypothesis,reference):
        scores = self.rouge.score(reference,hypothesis)
        return{
            'rouge1':scores['rouge1'].fmeasure,
            'rouge2':scores['rouge2'].fmeasure,
            'rougeL':scores['rougeL'].fmeasure
        }
    
    def bert_eval(self,hypothesis,reference):
        P,R,F1 = bert_score([hypothesis],[reference],lang='en')
        return {
            'precision':P.mean().item(),
            'recall':R.mean().item(),
            'f1':F1.mean().item()
        }
    
    def semantic_similarity(self, answer, reference):
        emb1 = self.st_model.encode(answer)
        emb2 = self.st_model.encode(reference)
        return util.cos_sim(emb1, emb2).item()
    
    def hallucination_rate(self, faithfulness_score):
        return round(1 - faithfulness_score, 4)
    
    def hit_rate(self, retrieved_contexts, ground_truth_context):
        """
        retrieved_contexts: list of chunks your retriever returned
        ground_truth_context: the chunk that actually contains the answer
        """
        for context in retrieved_contexts:
            if ground_truth_context.lower() in context.lower():
                return 1  # hit
        return 0  # miss


    def hit_rate_at_k(self, retrieved_contexts, ground_truth_context, k=3):
        """
        only checks top k retrieved chunks
        """
        for context in retrieved_contexts[:k]:
            if ground_truth_context.lower() in context.lower():
                return 1
        return 0


    def mean_hit_rate(self, all_retrieved, all_ground_truths, k=3):
        """
        run over entire test set and average
        """
        hits = [
            self.hit_rate_at_k(retrieved, gt, k)
            for retrieved, gt in zip(all_retrieved, all_ground_truths)
        ]
        return round(sum(hits) / len(hits), 4)

    def evaluate_all(self, question, answer, contexts, ground_truth, ground_truth_context=None, k=3):
        print("Running RAGAS...")
        ragas_results = self.ragas_eval(
            [question], [answer], [contexts], [ground_truth]
        )
        print("\n====================\nRunning ROUGE...")
        rouge_results = self.rouge_eval(answer, ground_truth)

        print("\n====================\nRunning BERTScore...")
        bert_results = self.bert_eval(answer, ground_truth)

        print("\n====================\nRunning Semantic Similarity...")
        similarity = self.semantic_similarity(answer, ground_truth)

        # get faithfulness score from ragas results
        faithfulness_score = ragas_results['faithfulness']
    
        print("Calculating Hallucination Rate...")
        hallucination = self.hallucination_rate(faithfulness_score)

        print("Calculating Hit Rate...")
        hit = self.hit_rate_at_k(contexts, ground_truth_context, k) if ground_truth_context else None

        return {
            "ragas": ragas_results,
            "rouge": rouge_results,
            "bert_score": bert_results,
            "semantic_similarity": similarity,
            "hallucination_rate": hallucination,
            "hit_rate": hit
        }
    

#     from rag_evaluator import RAGEvaluator

# evaluator = RAGEvaluator(groq_api_key="your_key")

# results = evaluator.evaluate_all(
#     question="What is RAG?",
#     answer="RAG is retrieval augmented generation...",
#     contexts=["chunk1 text", "chunk2 text"],
#     ground_truth="RAG stands for retrieval augmented generation..."
# )

# print(results)