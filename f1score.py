from torchmetrics.classification import BinaryF1Score
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def compute_f1(valid_queries, data, model, threshold):
    f1_sum = 0.0
    metric = BinaryF1Score(threshold=threshold).to(device)
    for q_index in range(valid_queries.q_count):
        query = valid_queries.queries[q_index]
        answers = valid_queries.answers[q_index]
        query = query.to(device)
        answers = answers.to(device)
        pred, _, _ = model(data, query)
        f1_sum += float(metric(pred, answers).cpu())
    
    # print(f1_sum, valid_queries.q_count, f1_sum / valid_queries.q_count)
    return f1_sum / valid_queries.q_count
