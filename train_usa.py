import torch
from datasets import load_dataset
import argparse
from tqdm import tqdm
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
import math
import gc 
import numpy as np

from USA import USA
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def load_data(name):
    
    if name == "billsum":
        dataset = load_dataset("FiscalNote/billsum")
    elif name == "qmsum":
        dataset = load_dataset("ioeddk/qmsum")
    else:
        raise NotImplementedError
    
    return dataset

def load_model(name):
    if name == "llama3-inst":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    else:
        raise NotImplementedError
    return model, tokenizer

def prompt_builder(model_name, data_name):
    
    if model_name == "llama3-inst" and data_name == "billsum":
        def builder (instance):
            messages = [{"role": "user", "content": 'Please summarize the following data: {}'.format(instance['text'])},
            {"role" : "assistant", "content": 'Tilte:{} \n Summary :{}'.format(instance['title'], instance['summary'])},
            {"role" : "user", "content" : 'Provide one line summary'}]
            return messages
        return builder
    elif model_name == "llama3-inst" and data_name == "qmsum":

        def builder (instance):
            messages = [{"role": "user", "content": '{}'.format(instance['text'].split("Answer:")[0])},
            {"role" : "assistant", "content": '{}'.format(instance['answer'])}]
            return messages
        return builder
    else:
        raise NotImplementedError


def compute_target_mask(attn_weights, mode):
    with torch.no_grad():
        if mode.startswith("top"):
            topk = int(mode.split('-')[1])
            mask = torch.zeros_like(attn_weights)
            _,idx = torch.topk(attn_weights, dim=-1, k=topk) # [B,A,S,T]
            view_idx = idx.view(-1,idx.shape[-1])
            view_idx = view_idx + torch.arange(view_idx.shape[0], device=view_idx.device).reshape(-1,1) * attn_weights.shape[-1]
            mask.view(-1)[view_idx.view(-1)] = 1.0
            return mask
        elif mode.startswith("coverage"):
            cov = float(mode.split('-')[1])
            mask = torch.zeros_like(attn_weights)
            values,idx = torch.sort(attn_weights, descending=True, dim=-1) # [B,A,S,T]
            values = (torch.cumsum(values, dim=-1) < cov)
            view_idx = idx.view(-1,idx.shape[-1])
            values_view = values.view(-1, values.shape[-1])
            view_idx = view_idx + torch.arange(view_idx.shape[0], device=view_idx.device).reshape(-1,1) * attn_weights.shape[-1]
            mask.view(-1)[view_idx.view(-1)[values_view.view(-1)]] = 1.0
            return mask
        else:
            raise NotImplementedError


def evaluate_coverage(model, usas, test_data_loader, num_samples=10):
    coverages = torch.zeros(len(usas), usas[0].num_heads, device="cuda:0")
    sparsities = torch.zeros(len(usas), usas[0].num_heads, device="cuda:0")
    num = 0
    for (batch_idx, batch) in tqdm(enumerate(test_data_loader), total=min(num_samples, test_data_loader.__len__())):
        if batch_idx > num_samples:
            break
        with torch.no_grad():        
            output_dict = model(input_ids = batch['input_ids'].to("cuda:1"),
                    attention_mask = batch['attention_mask'].to("cuda:1"),
                    past_key_values=DynamicCache(),
                    use_cache=True,
                    output_attentions=False,
                    return_dict=True)

        total_loss = 0
        losses = []
        for i in range(len(usas)):
            usa = usas[i].to("cuda:0")
            K = output_dict['past_key_values'].key_cache[i].detach().to("cuda:0")
            Q = output_dict['past_key_values'].query_cache[i].detach().to("cuda:0")
            Q = Q[:,:,Q.shape[2]//2:,:]
            K = repeat_kv(K, Q.shape[1] // K.shape[1])
            sspan = usa(K, Q, hard=True)
            target = torch.nn.functional.softmax(torch.matmul(Q, K.transpose(2,3)) / math.sqrt(K.shape[-1]), dim=-1, dtype=torch.float32).to(Q.dtype)
            coverage = torch.sum(target * sspan, dim=-1).mean(dim=-1).mean(dim=0)
            coverages[i] += coverage
            
            sparsity = sspan.mean(dim=-1).mean(dim=-1).mean(dim=0)
            sparsities[i] += sparsity

        num+=1

    coverages = coverages / num
    sparsities = sparsities / num
    print("--- coverage ---")
    print(coverages)
    print("--- sparsities ---")
    print(sparsities)
    return {'coverage' : torch.mean(coverages).item(),
            'sparsity' : torch.mean(sparsities).item()}



def train_step(usa, K, Q, target_mode, span_criterion):
    # GQA
    K = repeat_kv(K, Q.shape[1] // K.shape[1])
    Q = Q[:,:,Q.shape[2]//2:,:] # only use last half of queries
    output = usa(K, Q)
    attention_scores = torch.nn.functional.softmax(torch.matmul(Q, K.transpose(2,3)) / math.sqrt(K.shape[-1]), dim=-1, dtype=torch.float32).to(Q.dtype)
    target = compute_target_mask(attention_scores, target_mode)
    span_loss = span_criterion(torch.nn.functional.relu(output), target)
    return span_loss

def train_usa(model, usas, train_data_loader, test_data_loader, target_mode, optimizer, span_criterion, max_train_itr):
    itr = 0
    for (batch_idx, batch) in tqdm(enumerate(train_data_loader), total=train_data_loader.__len__()):
            itr += 1
            if max_train_itr > 0 and itr > max_train_itr:
                break
            if batch_idx % 10 == 0:
                result = evaluate_coverage(model, usas, test_data_loader, num_samples=10)
                print('{}/{} coverage:{:.6f} sparsity:{:.6f}'.format(batch_idx, train_data_loader.__len__(), result['coverage'], result['sparsity']))

                torch.save(usas.cpu().state_dict(), "./artifacts/usa.pt")

            with torch.no_grad():        
                output_dict = model(input_ids = batch['input_ids'].to("cuda:1"),
                        attention_mask = batch['attention_mask'].to("cuda:1"),
                        past_key_values=DynamicCache(),
                        use_cache=True,
                        output_attentions=True,
                        return_dict=True)

            total_loss = 0
            losses = []
            for i in range(len(usas)):
                usa = usas[i].to("cuda:0")
                K = output_dict['past_key_values'].key_cache[i].detach().to("cuda:0")
                Q = output_dict['past_key_values'].query_cache[i].detach().to("cuda:0")
                loss = train_step(usa, K, Q, target_mode, span_criterion)
                total_loss += loss
                losses.append(loss.item())
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            print(np.mean(losses), losses, flush=True)
            


def data_pipeline(train_data, tokenizer, builder, max_len):
    chat_train_data = train_data.map(lambda instance: {'prompt' : tokenizer.apply_chat_template(builder(instance), tokenize=False, add_generation_prompt=True) })
    encoded_train_data = chat_train_data.map(lambda instance: tokenizer(instance['prompt']))    
    def middle_truncate(instance, length=max_len):
        assert(len(instance['attention_mask']) >= length) #TODO(pad the smaller sequences)
        mid_point = len(instance['attention_mask']) // 2
        excess = len(instance['attention_mask']) - length
        left = mid_point - excess // 2
        return {'input_ids' : instance['input_ids'][:left] + instance['input_ids'][left + excess:],
                'attention_mask' : instance['attention_mask'][:left] + instance['attention_mask'][left + excess:],
                }
    truncated_encoded_train_data = encoded_train_data.map(middle_truncate)
    truncated_encoded_train_data.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    return truncated_encoded_train_data
    
def main():
    parser = argparse.ArgumentParser(description=" Train Script for USA")

    # Add arguments
    parser.add_argument("--dataset", type=str, help="dataset", choices=["billsum", "qmsum"])
    parser.add_argument("--model", type=str, help="model", choices=["llama3-inst"])
    parser.add_argument("--train_epochs", type=int, help="train args: epochs", default=1)
    parser.add_argument("--batch", type=int, help="train args: batchsize", default=1)
    parser.add_argument("--max_length", type=int, help="max length", default=512)
    parser.add_argument("--max_train_itr", type=int, help="max train itr", default=-1)
    parser.add_argument("--device", type=str, help="train args: device", default="cuda")
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")

    # usa 
    parser.add_argument("--usa-load", type=str, help="load usa", default=None)
    parser.add_argument("--usa-L", type=int, help="usa L", default=32)
    parser.add_argument("--usa-R", type=int, help="usa R", default=4)
    parser.add_argument("--usa-int-dim", type=int, help="usa int dim", default=256)
    parser.add_argument("--usa-aug-k", type=int, help="usa aug k", default=0)
    parser.add_argument("--usa-target-mode", type=str, help="usa aug k", default="top-10")


    args = parser.parse_args()

    assert(torch.cuda.device_count() >= 2)
    # we will put the model inference on device 1 and perform training usas on device 0

    ## Model and prompt stuff
    model, tokenizer = load_model(args.model)
    model = model.to("cuda:1")
    tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = "left"
    builder = prompt_builder(args.model, args.dataset)
    
    ## dataset loading
    dataset = load_data(args.dataset)
    train_data = dataset['train']
    test_data = dataset['test']

    truncated_encoded_train_data = data_pipeline(train_data, tokenizer, builder, args.max_length)
    truncated_encoded_test_data = data_pipeline(test_data, tokenizer, builder, args.max_length)

    # chat_train_data = train_data.map(lambda instance: {'prompt' : tokenizer.apply_chat_template(builder(instance), tokenize=False, add_generation_prompt=True) })
    # encoded_train_data = chat_train_data.map(lambda instance: tokenizer(instance['prompt']))    
    # def middle_truncate(instance, length=args.max_length):
    #     assert(len(instance['attention_mask']) >= length) #TODO(pad the smaller sequences)
    #     mid_point = len(instance['attention_mask']) // 2
    #     excess = len(instance['attention_mask']) - length
    #     left = mid_point - excess // 2
    #     return {'input_ids' : instance['input_ids'][:left] + instance['input_ids'][left + excess:],
    #             'attention_mask' : instance['attention_mask'][:left] + instance['attention_mask'][left + excess:],
    #             }
    # truncated_encoded_train_data = encoded_train_data.map(middle_truncate)
    # truncated_encoded_train_data.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    train_data_loader = torch.utils.data.DataLoader(truncated_encoded_train_data, batch_size=args.batch)
    test_data_loader = torch.utils.data.DataLoader(truncated_encoded_test_data, batch_size=args.batch)


    ## USA loading

    usas = []
    for i in range(len(model.model.layers)):
        usas.append( USA( model.model.layers[0].self_attn.num_heads, 
                        model.model.layers[i].self_attn.head_dim,
                        usa_params = {'L': args.usa_L, 'R': args.usa_R, 'int_dim': args.usa_int_dim, 'aug_k' : 0},
                        annealing_paramters = {'T': 10, 't': 0}
                        )
                    )
    usas = torch.nn.ModuleList(usas).to("cuda:0")
    if args.usa_load is not None:
        usas.load_state_dict(torch.load(args.usa_load))
    
    span_criterion = torch.nn.BCELoss() # Define your loss function (e.g., torch.nn.CrossEntropyLoss())
    optimizer = torch.optim.Adam(usas.parameters()) # Define your optimizer (e.g., torch.optim.Adam(model.parameters()))

    train_usa(model, usas, train_data_loader, test_data_loader, args.usa_target_mode, optimizer, span_criterion, args.max_train_itr)
    torch.save(usas.cpu().state_dict(), "./artifacts/usa-" + args.model + "-config-" + 'L_{}-R_{}-I_{}-mode-{}'.format(args.usa_L, args.usa_R, args.usa_int_dim, args.usa_target_mode))

if __name__ == '__main__':
    main()
