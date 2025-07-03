import torch
from tqdm import tqdm

from TransformerLens.transformer_lens import HookedEncoder, ActivationCache
from TransformerLens.transformer_lens import patching
import transformer_lens.utils as utils

from jaxtyping import Float
from typing import Callable
from functools import partial
import itertools

'''
    Patches a given sequence position in the residual stream, using the value
    from the clean cache.
'''
def patch_residual_component(
    corrupted_component: Float[torch.Tensor, "batch pos d_model"],
    hook,
    pos,
    clean_cache,
):
    corrupted_component[:, pos, :] = clean_cache[hook.name][:, pos, :]
    return corrupted_component


'''
Returns an array of results of patching each position at each layer in the residual
stream, using the value from the clean cache.

The results are calculated using the patching_metric function, which should be
called on the model's logit output.
'''
def get_act_patch_block_every(
    model: HookedEncoder, 
    device,
    corrupted_tokens: Float[torch.Tensor, "batch pos"], 
    clean_cache: ActivationCache, 
    patching_metric: Callable[[Float[torch.Tensor, "batch pos d_vocab"]], float],
    pooler_layer,
    dropout_layer,
    classifier_layer,
) -> Float[torch.Tensor, "3 layer pos"]:

    model.reset_hooks()
    _, seq_len = corrupted_tokens["input_ids"].size()
    results = torch.zeros(3, model.cfg.n_layers, seq_len, device=device, dtype=torch.float32)

    # send tokens to device if not already there
    corrupted_tokens["input_ids"] = corrupted_tokens["input_ids"].to(device)
    corrupted_tokens["attention_mask"] = corrupted_tokens["attention_mask"].to(device)

    for component_idx, component in enumerate(["resid_pre", "attn_out", "mlp_out"]):
        print("Patching:", component)
        for layer in tqdm(range(model.cfg.n_layers)):
            for position in range(seq_len):
                hook_fn = partial(patch_residual_component, pos=position, clean_cache=clean_cache)
                patched_outputs = model.run_with_hooks(
                    corrupted_tokens["input_ids"],
                    one_zero_attention_mask=corrupted_tokens["attention_mask"],
                    token_type_ids = corrupted_tokens['token_type_ids'],
                    return_type="embeddings",
                    fwd_hooks = [(utils.get_act_name(component, layer), hook_fn)],
                )
                #patched_embedding = patched_outputs[:,0,:].squeeze(0)#JENNIFER
                results[component_idx, layer, position] = patching_metric(patched_outputs,pooler_layer,dropout_layer,classifier_layer)
    return results

def get_act_patch_block_every_with_ablated_dup_token_heads(
    model: HookedEncoder, 
    device,
    corrupted_tokens: Float[torch.Tensor, "batch pos"], 
    clean_cache: ActivationCache, 
    patching_metric: Callable[[Float[torch.Tensor, "batch pos d_vocab"]], float],
    pooler_layer,
    dropout_layer,
    classifier_layer,
    ABLATED_HEADS =[(1,7),(5,9),(7,9), (8,5), (8,3),(8,0), (3,1), (2,1)]+[(7,8),(4,9),(7,2),(7,4),(7,3)]
) -> Float[torch.Tensor, "3 layer pos"]:

    
    _, seq_len = corrupted_tokens["input_ids"].size()
    results = torch.zeros(3, model.cfg.n_layers, seq_len, device=device, dtype=torch.float32)

    # send tokens to device if not already there
    corrupted_tokens["input_ids"] = corrupted_tokens["input_ids"].to(device)
    corrupted_tokens["attention_mask"] = corrupted_tokens["attention_mask"].to(device)
    


    for component_idx, component in enumerate(["resid_pre", "attn_out", "mlp_out"]):
        print("Patching:", component)
        for layer in tqdm(range(model.cfg.n_layers)):
            for position in range(seq_len):
                model.reset_hooks()
                act_name_filter = lambda name: name in [utils.get_act_name("z", layer) for layer, _ in ABLATED_HEADS]
                ablate_fn = partial(head_ablation_hook, heads_to_ablate=ABLATED_HEADS)
                model.add_hook(act_name_filter, ablate_fn,level=1)
                hook_fn = partial(patch_residual_component, pos=position, clean_cache=clean_cache)
                model.add_hook(utils.get_act_name(component, layer), hook_fn,level=1)

                patched_outputs,_ = model.run_with_cache(
                    corrupted_tokens["input_ids"],
                    one_zero_attention_mask=corrupted_tokens["attention_mask"],
                    token_type_ids = corrupted_tokens['token_type_ids'],
                    return_type="embeddings",
                )
                #patched_embedding = patched_outputs[:,0,:].squeeze(0)#JENNIFER
                results[component_idx, layer, position] = patching_metric(patched_outputs,pooler_layer,dropout_layer,classifier_layer)
 
    return results

   



'''
Patches the output of a given head (before it's added to the residual stream) at
every sequence position, using the value from the clean cache.
'''
def patch_head_vector(
    corrupted_head_vector: Float[torch.Tensor, "batch pos head_index d_head"],
    hook, #: HookPoint, 
    head_index: int, 
    clean_cache: ActivationCache
) -> Float[torch.Tensor, "batch pos head_index d_head"]:
 
    corrupted_head_vector[:, :, head_index] = clean_cache[hook.name][:, :, head_index]
    return corrupted_head_vector

def patch_mlp_vector(
    corrupted_head_vector: Float[torch.Tensor, "batch pos d_model"],
    hook, #: HookPoint, 
    clean_cache: ActivationCache
) -> Float[torch.Tensor, "batch pos d_model"]:
    corrupted_head_vector = clean_cache[hook.name]
    return corrupted_head_vector


'''
Returns an array of results of patching at all positions for each head in each
layer, using the value from the clean cache.

The results are calculated using the patching_metric function, which should be
called on the model's embedding output.
'''
def get_act_patch_attn_head_out_all_pos( #TODO: v is not attn_head_out right?
    model: HookedEncoder, 
    device,
    corrupted_tokens: Float[torch.Tensor, "batch pos"], 
    clean_cache: ActivationCache, 
    patching_metric: Callable,
    pooler_layer,
    dropout_layer,
    classifier_layer
) -> Float[torch.Tensor, "layer head"]:

    model.reset_hooks()
    results = torch.zeros(model.cfg.n_layers, model.cfg.n_heads, device=device, dtype=torch.float32)
    layer_head_list=[(1,7),(5,9),(7,9), (8,5), (8,3),(8,0), (3,1), (2,1)]

    print("Patching: attn_heads")
    for layer in tqdm(range(model.cfg.n_layers)):
        for head in range(model.cfg.n_heads):
            
            hook_fn = partial(patch_head_vector, head_index=head, clean_cache=clean_cache)
            patched_outputs = model.run_with_hooks(
                corrupted_tokens["input_ids"],
                one_zero_attention_mask=corrupted_tokens["attention_mask"],
                token_type_ids = corrupted_tokens['token_type_ids'],
                return_type="embeddings",
                fwd_hooks = [(utils.get_act_name("v", layer), hook_fn)],
            )
            #print(classifier_layer(dropout_layer(pooler_layer(patched_outputs))))

            #patched_embedding = patched_outputs[:,0,:].squeeze(0) #JENNIFER
            results[layer, head] = patching_metric(patched_outputs,pooler_layer,dropout_layer,classifier_layer)
            #if (layer,head) in layer_head_list:
            #    print(layer,head,results[layer, head])
                #print(results[layer, head])

    return results

def get_act_patch_attn_head_z_all_pos( 
    model: HookedEncoder, 
    device,
    corrupted_tokens: Float[torch.Tensor, "batch pos"], 
    clean_cache: ActivationCache, 
    patching_metric: Callable,
    pooler_layer,
    dropout_layer,
    classifier_layer
) -> Float[torch.Tensor, "layer head"]:

    model.reset_hooks()
    results = torch.zeros(model.cfg.n_layers, model.cfg.n_heads, device=device, dtype=torch.float32)

    print("Patching: attn_heads")
    for layer in tqdm(range(model.cfg.n_layers)):
        for head in range(model.cfg.n_heads):
            hook_fn = partial(patch_head_vector, head_index=head, clean_cache=clean_cache)
            patched_outputs = model.run_with_hooks(
                corrupted_tokens["input_ids"],
                one_zero_attention_mask=corrupted_tokens["attention_mask"],
                token_type_ids = corrupted_tokens['token_type_ids'],
                return_type="embeddings",
                fwd_hooks = [(utils.get_act_name("z", layer), hook_fn)],
            )
            #patched_embedding = patched_outputs[:,0,:].squeeze(0) #JENNIFER
            results[layer, head] = patching_metric(patched_outputs,pooler_layer,dropout_layer,classifier_layer)

    return results






def patch_head_vector_by_pos_pattern(
    corrupted_activation: Float[torch.Tensor, "batch pos head_index pos_q pos_k"],
    hook, #: HookPoint, 
    pos,
    head_index: int, 
    clean_cache: ActivationCache
) -> Float[torch.Tensor, "batch pos head_index d_head"]:

    corrupted_activation[:,head_index,pos,:] = clean_cache[hook.name][:,head_index,pos,:]
    return corrupted_activation


def patch_head_vector_by_pos(
    corrupted_activation: Float[torch.Tensor, "batch pos head_index d_head"],
    hook, #: HookPoint, 
    pos,
    head_index: int, 
    clean_cache: ActivationCache
) -> Float[torch.Tensor, "batch pos head_index d_head"]:

    corrupted_activation[:, pos, head_index] = clean_cache[hook.name][:, pos, head_index]
    return corrupted_activation


def get_act_patch_attn_head_by_pos(
    model: HookedEncoder, 
    device,
    corrupted_tokens: Float[torch.Tensor, "batch pos"], 
    clean_cache: ActivationCache, 
    patching_metric: Callable,
    layer_head_list,
    pooler_layer,
    dropout_layer,
    classifier_layer
    
) -> Float[torch.Tensor, "layer pos head"]:
    
    model.reset_hooks()
    _, seq_len = corrupted_tokens["input_ids"].size()
    results = torch.zeros(2, len(layer_head_list), seq_len, device=device, dtype=torch.float32)

    for component_idx, component in enumerate(["z", "pattern"]):
        for i, layer_head in enumerate(layer_head_list):
            layer = layer_head[0]
            head = layer_head[1]
            for position in range(seq_len):
                patch_fn = patch_head_vector_by_pos_pattern if component == "pattern" else patch_head_vector_by_pos
                hook_fn = partial(patch_fn, pos=position, head_index=head, clean_cache=clean_cache)
                patched_outputs = model.run_with_hooks(
                    corrupted_tokens["input_ids"],
                    one_zero_attention_mask=corrupted_tokens["attention_mask"],
                    token_type_ids = corrupted_tokens['token_type_ids'],
                    return_type="embeddings",
                    fwd_hooks = [(utils.get_act_name(component, layer), hook_fn)],
                )
                #patched_embedding = patched_outputs[:,0,:].squeeze(0)
                results[component_idx, i, position] = patching_metric(patched_outputs,pooler_layer,dropout_layer,classifier_layer)

    return results

def patch_or_freeze_head_vectors( #freeze everything in baseline cache, but patch in the sender component from the perturbed cache
    orig_head_vector,# Float[Tensor, "batch pos head_index d_head"], #sending everything that ends in z #for each layer, you patch the sender head
    hook,#: HookPoint,
    perturbed_cache: ActivationCache,
    baseline_cache: ActivationCache, #orig_cache, patch in 
    head_to_patch#: Tuple[int, int],
):# -> Float[Tensor, "batch pos head_index d_head"]:
    '''
    This helps implement step 2 of path patching. We freeze all head outputs (i.e. set them
    to their values in orig_cache), except for head_to_patch (if it's in this layer) which
    we patch with the value from new_cache.

    head_to_patch: tuple of (layer, head)
        we can use hook.layer() to check if the head to patch is in this layer
    '''
    #orig_cache contains only the z (head output values), so here we are freezing all z values except head_to_patch
    orig_head_vector[...] = baseline_cache[hook.name][...] # set all heads (z) with orig_cache values, the ellipsis (...) is used as a shorthand for selecting all elements in all dimensions.
    if head_to_patch[0] == hook.layer(): #if the sender is not the hook.layer(), then don't change anything
        #sender 
        orig_head_vector[:, :, head_to_patch[1]] = perturbed_cache[hook.name][:, :, head_to_patch[1]] #patch in for the receiver head from the new_cache
    return orig_head_vector
    #if head_to_patch is not in the layer that was sent to the hook, don't change anything to the z, then it means that this layer's z would stay the same (frozen)


def get_path_patch_head_to_final_resid_post(
    model: HookedEncoder, 
    device,
    patching_metric: Callable,
    perturbed_tokens,#: IOIDataset = abc_dataset,
    baseline_tokens,#: IOIDataset = ioi_dataset,
    perturbed_cache,#: Optional[ActivationCache] = abc_cache,
    baseline_cache,#: Optional[ActivationCache] = ioi_cache,
    pooler_layer,
    dropout_layer,
    classifier_layer
): #-> Float[Tensor, "layer head"]:
    '''
    Performs path patching (see algorithm in appendix B of IOI paper), with:
        sender head = (each head, looped through, one at a time)
        receiver node = final value of residual stream
    Returns:
        tensor of metric values for every possible sender head
    '''
    model.reset_hooks()
    results = torch.zeros(model.cfg.n_layers, model.cfg.n_heads, device=device, dtype=torch.float32)
    resid_post_hook_name = utils.get_act_name("resid_post", model.cfg.n_layers - 1)
    resid_post_name_filter = lambda name: name == resid_post_hook_name
    # ========== Step 1 ==========
    # Gather activations on x_orig and x_new
    z_name_filter = lambda name: name.endswith("z") #'blocks.0.attn.z' for all layers
    #print(z_name_filter)
    if perturbed_cache is None:
        _, perturbed_cache = model.run_with_cache(
            perturbed_tokens["input_ids"],
            one_zero_attention_mask=perturbed_tokens["attention_mask"],
            token_type_ids = perturbed_tokens['token_type_ids'],
            names_filter=z_name_filter, #we only cache the things we need (in this case, just attn head outputs).
            return_type=None
        )
    if baseline_cache is None:
        _, baseline_cache = model.run_with_cache(
            baseline_tokens["input_ids"],
            one_zero_attention_mask=baseline_tokens["attention_mask"],
            token_type_ids = baseline_tokens['token_type_ids'],
            names_filter=z_name_filter,
            return_type=None
        )
    # Looping over every possible sender head (the receiver is always the final resid_post)
    # Note use of itertools (gives us a smoother progress bar)
    for (sender_layer, sender_head) in tqdm(list(itertools.product(range(model.cfg.n_layers), range(model.cfg.n_heads)))):
        """
        #for every possible sender node, we update the hook function to patch the specific sender node, and send in multiple hooks that hook to all z in each layer ('blocks.0.attn.z')
        #the hook function checks whether the sender node is in the layer, if so patch that head and keeps everything else the same; if not keeps everything the same in the layer
        """

        # ========== Step 2 ==========
        # Run on x_orig, with sender head patched from x_new, every other head frozen
        hook_fn = partial( #partial() means you are updating the 3rd-last parameters of the hook function to create some variations
            patch_or_freeze_head_vectors, #hook_fn name
            perturbed_cache=perturbed_cache, #3rd-last params #abc
            baseline_cache=baseline_cache,#ioi
            head_to_patch=(sender_layer, sender_head), #all different combinations
        )
        model.add_hook(z_name_filter, hook_fn,level=1)
        #model.add_hook(z_name_filter, hook_fn)
        #add_hook similar to run_with_hook, will send the first argument (z_name_filter)
        #means that: sending in every layer every head's z
        #for both add_hook and run_with_hook: lambda name: name.endswith("pattern")

        #instead of using model.run_with_hooks, using  model.add_hook + model.run_with_cache
        patched_outputs, patched_cache = model.run_with_cache(
            baseline_tokens['input_ids'],#running with clean tokens, patch in a perturbed sender component 
            one_zero_attention_mask=baseline_tokens["attention_mask"],
            token_type_ids = baseline_tokens['token_type_ids'],
            names_filter=resid_post_name_filter,#the receiver is always the final resid_post = here we mean only saved the cache for the final_resid_post
            return_type="embeddings"
        )
        assert set(patched_cache.keys()) == {resid_post_hook_name}
        # ========== Step 3 ==========
        # Unembed the final residual stream value, to get our patched logits
        # ORIGINAL CODE: patched_logits = model.unembed(model.ln_final(patched_cache[resid_post_hook_name]))
        #NOTE: patched_cache is hook_resid_pre[11], patched_outputs is hook_normalized_resid_post[11] 
        #so instead of normalizing ourselves, we can just use patched_outputs

        # Save the results
        #results[sender_layer, sender_head] = patching_metric(patched_logits)
        results[sender_layer, sender_head] = patching_metric(patched_outputs,pooler_layer,dropout_layer,classifier_layer)

    return results

def patch_head_input(
    orig_activation,#: Float[Tensor, "batch pos head_idx d_head"],
    hook,#: HookPoint,
    patched_cache,#: ActivationCache,
    head_list,#: List[Tuple[int, int]],
): #-> Float[Tensor, "batch pos head_idx d_head"]:
    '''
    Function which can patch any combination of heads in layers,
    according to the heads in head_list. 
    Can patch multiple heads simultaneously if when calling the patch_head_input pass in a list of hook names
    such as this example:
            patched_outputs = model.run_with_hooks(
            baseline_tokens["input_ids"],
            fwd_hooks = [(receiver_hook_names_filter, hook_fn)],
            return_type="embeddings"
        )

    '''
    heads_to_patch = [head for layer, head in head_list if layer == hook.layer()]
    orig_activation[:, :, heads_to_patch] = patched_cache[hook.name][:, :, heads_to_patch]
    return orig_activation


def get_path_patch_head_to_heads(
    receiver_heads,#,: List[Tuple[int, int]],
    receiver_input,#: str,
    model: HookedEncoder,
    device,
    patching_metric: Callable,
    perturbed_tokens,
    baseline_tokens,
    perturbed_cache,
    baseline_cache,
    pooler_layer,
    dropout_layer,
    classifier_layer
):
    '''
    Performs path patching (see algorithm in appendix B of IOI paper), with:

        sender head = (each head, looped through, one at a time)
        receiver node = input to a later head (or set of heads)

    The receiver node is specified by receiver_heads and receiver_input.
    Example (for S-inhibition path patching the queries):
        receiver_heads = [(8, 6), (8, 10), (7, 9), (7, 3)],
        receiver_input = "v"

    Returns:
        tensor of metric values for every possible sender head
    '''
    # SOLUTION
    model.reset_hooks()

    assert receiver_input in ("k", "q", "v")#NOTE 
    receiver_layers = set(next(zip(*receiver_heads)))
    receiver_hook_names = [utils.get_act_name(receiver_input, layer) for layer in receiver_layers]
    receiver_hook_names_filter = lambda name: name in receiver_hook_names

    results = torch.zeros(max(receiver_layers), model.cfg.n_heads, device=device, dtype=torch.float32)

    # ========== Step 1 ==========
    # Gather activations on x_orig and x_new

    # Note the use of names_filter for the run_with_cache function. Using it means we
    # only cache the things we need (in this case, just attn head outputs).
    z_name_filter = lambda name: name.endswith("z") #'blocks.0.attn.z' for all layers
    #print(z_name_filter)
    if perturbed_cache is None:
        _, perturbed_cache = model.run_with_cache(
            perturbed_tokens["input_ids"],
            one_zero_attention_mask=perturbed_tokens["attention_mask"],
            token_type_ids = perturbed_tokens['token_type_ids'],
            names_filter=z_name_filter, #we only cache the things we need (in this case, just attn head outputs).
            return_type=None
        )
    if baseline_cache is None:
        _, baseline_cache = model.run_with_cache(
            baseline_tokens["input_ids"],
            one_zero_attention_mask=baseline_tokens["attention_mask"],
            token_type_ids = baseline_tokens['token_type_ids'],
            names_filter=z_name_filter,
            return_type=None
        )

    # Note, the sender layer will always be before the final receiver layer, otherwise there will
    # be no causal effect from sender -> receiver. So we only need to loop this far.
    for (sender_layer, sender_head) in tqdm(list(itertools.product(
        range(max(receiver_layers)),
        range(model.cfg.n_heads)
    ))):

        # ========== Step 2 ==========
        # Run on x_orig, with sender head patched from x_new, every other head frozen

        hook_fn = partial( #partial() means you are updating the 3rd-last parameters of the hook function to create some variations
            patch_or_freeze_head_vectors, #hook_fn name
            perturbed_cache=perturbed_cache, #3rd-last params
            baseline_cache=baseline_cache,
            head_to_patch=(sender_layer, sender_head), #all different combinations
        )
        model.add_hook(z_name_filter, hook_fn,level=1)

        patched_outputs, patched_cache = model.run_with_cache(
            baseline_tokens['input_ids'],#running with clean tokens, patch in a perturbed sender component 
            one_zero_attention_mask=baseline_tokens["attention_mask"],
            token_type_ids = baseline_tokens['token_type_ids'],
            names_filter=receiver_hook_names_filter,#the receiver is always the final resid_post = here we mean only saved the cache for the final_resid_post
            return_type='embeddings'
        )
        score = classifier_layer(pooler_layer(dropout_layer(patched_outputs)))
        print(score)
        # model.reset_hooks(including_permanent=True)
        assert set(patched_cache.keys()) == set(receiver_hook_names)

        # ========== Step 3 ==========
        # Run on x_orig, patching in the receiver node(s) from the previously cached value

        hook_fn = partial(
            patch_head_input,
            patched_cache=patched_cache,#patched cache has all other heads frozen, just the sender heads importance on the receivers
                                        # since your receiver nodes are in the middle of the model rather than at the very end, you will have to run the model again 
                                        # with these nodes patched in rather than just calculating the logit output directly from the patched values of the final residual stream.
                                        #Why can't we earlier just return the logits - because the earlier hookedfunction just runs until the receivers
                                        #also for step 3 don't need to freeze, because earlier has already isolated the direct influence 
            head_list=receiver_heads,
        )
        patched_outputs = model.run_with_hooks(
            baseline_tokens["input_ids"],
            one_zero_attention_mask=baseline_tokens["attention_mask"],
            token_type_ids = baseline_tokens['token_type_ids'],
            fwd_hooks = [(receiver_hook_names_filter, hook_fn)],
            return_type="embeddings"
        )

        # Save the results
        #results[layer, head] = patching_metric(patched_outputs,pooler_layer,dropout_layer,classifier_layer)
        score = classifier_layer(pooler_layer(dropout_layer(patched_outputs)))
        #print(score)
        
        results[sender_layer, sender_head] = patching_metric(patched_outputs,pooler_layer,dropout_layer,classifier_layer)

    return results#TODO: need to not pass out the score

def get_path_patch_components_to_heads(
    receiver_heads,#,: List[Tuple[int, int]],
    receiver_input,#: str,
    model: HookedEncoder,
    device,
    patching_metric: Callable,
    perturbed_tokens,
    baseline_tokens,
    perturbed_cache,
    baseline_cache,
    pooler_layer,
    dropout_layer,
    classifier_layer
):
    model.reset_hooks()

    _, seq_len = baseline_tokens["input_ids"].size()
    assert receiver_input in ("k", "q", "v")#NOTE 
    receiver_layers = set(next(zip(*receiver_heads)))
    receiver_hook_names = [utils.get_act_name(receiver_input, layer) for layer in receiver_layers]
    receiver_hook_names_filter = lambda name: name in receiver_hook_names

    results = torch.zeros(3,max(receiver_layers), seq_len, device=device, dtype=torch.float32)

    # ========== Step 1 ==========
    # Gather activations on x_orig and x_new
    # Note the use of names_filter for the run_with_cache function. Using it means we
    # only cache the things we need (in this case, just attn head outputs).
    #z_name_filter = lambda name: name.endswith("z") #'blocks.0.attn.z' for all layers
    # Note, the sender layer will always be before the final receiver layer, otherwise there will
    # be no causal effect from sender -> receiver. So we only need to loop this far.
    for component_idx, component in enumerate(["resid_pre", "attn_out", "mlp_out"]):
        for layer in range(max(receiver_layers)):
            for position in range(seq_len):
                z_name_filter = lambda name: name in [utils.get_act_name(component,layer)]
                #print([utils.get_act_name(component,layer)])
                hook_fn = partial(patch_residual_component, pos=position, clean_cache=perturbed_cache)
    
                # ========== Step 2 ==========
                # Run on x_orig, with sender head patched from x_new, every other head frozen

                model.add_hook(z_name_filter, hook_fn,level=1)

                patched_outputs, patched_cache = model.run_with_cache(
                    baseline_tokens['input_ids'],#running with clean tokens, patch in a perturbed sender component 
                    one_zero_attention_mask=baseline_tokens["attention_mask"],
                    token_type_ids = baseline_tokens['token_type_ids'],
                    names_filter=receiver_hook_names_filter,#the receiver is always the final resid_post = here we mean only saved the cache for the final_resid_post
                    return_type='embeddings'
                )
                score = classifier_layer(pooler_layer(dropout_layer(patched_outputs)))
                #print(score)
                # model.reset_hooks(including_permanent=True)
                assert set(patched_cache.keys()) == set(receiver_hook_names)

                # ========== Step 3 ==========
                model.reset_hooks()
                # Run on x_orig, patching in the receiver node(s) from the previously cached value
                model.add_hook(z_name_filter, hook_fn,level=1)
                hook_fn = partial(
                    patch_head_input,
                    patched_cache=patched_cache,#patched cache has all other heads frozen, just the sender heads importance on the receivers
                                                # since your receiver nodes are in the middle of the model rather than at the very end, you will have to run the model again 
                                                # with these nodes patched in rather than just calculating the logit output directly from the patched values of the final residual stream.
                                                #Why can't we earlier just return the logits - because the earlier hookedfunction just runs until the receivers
                                                #also for step 3 don't need to freeze, because earlier has already isolated the direct influence 
                    head_list=receiver_heads,
                )
                new_patched_outputs = model.run_with_hooks(
                    baseline_tokens["input_ids"],
                    one_zero_attention_mask=baseline_tokens["attention_mask"],
                    token_type_ids = baseline_tokens['token_type_ids'],
                    fwd_hooks = [(receiver_hook_names_filter, hook_fn)],
                    return_type="embeddings"
                )
                score = classifier_layer(pooler_layer(dropout_layer(new_patched_outputs)))
                #print(score)
                results[component_idx,layer, position] = patching_metric(new_patched_outputs,pooler_layer,dropout_layer,classifier_layer)
                print(results[component_idx,layer, position].item())
    return results#TODO: need to not pass out the score
#this function pathces to ONE specified head's z and see the logit difference #used by term-freq-info experiment
def patch_specific_head_attn_z_all_pos(
    model: HookedEncoder, 
    device,
    corrupted_tokens: Float[torch.Tensor, "batch pos"], 
    clean_cache: ActivationCache, 
    patching_metric: Callable,
    pooler_layer,
    dropout_layer,
    classifier_layer,
    layer_head, #specific layer_head to patch
) -> Float[torch.Tensor, "layer head"]:

    model.reset_hooks()
    layer,head = layer_head[0],layer_head[1]
    hook_fn = partial(patch_head_vector, head_index=head, clean_cache=clean_cache) #because the passed-in clean_cache is actually perturbed_cache
    patched_outputs = model.run_with_hooks(
        corrupted_tokens["input_ids"],
        one_zero_attention_mask=corrupted_tokens["attention_mask"],
        token_type_ids = corrupted_tokens['token_type_ids'],
        return_type="embeddings",
        fwd_hooks = [(utils.get_act_name("z", layer), hook_fn)],
    )
    #patched_embedding = patched_outputs[:,0,:].squeeze(0) #JENNIFER
    logit_diff = patching_metric(patched_outputs,pooler_layer,dropout_layer,classifier_layer)

    return logit_diff

#this function pathces to specified heads'z (a list) together and see the logit difference #used by term-freq-info experiment
def patch_specific_multi_heads_attn_z_all_pos(
    model: HookedEncoder, 
    device,
    corrupted_tokens: Float[torch.Tensor, "batch pos"], #when passed in should be the clean tokens
    clean_cache: ActivationCache, #when passed in should be ht eperturbed cache
    patching_metric: Callable,
    pooler_layer,
    dropout_layer,
    classifier_layer,
    heads_to_patch_list, #specific layer_head to patch
    
) -> Float[torch.Tensor, "layer head"]:
    model.reset_hooks()
    layers = set(next(zip(*heads_to_patch_list)))
    hook_names = [utils.get_act_name('z', layer) for layer in layers]
    hook_names_filter = lambda name: name in hook_names
    #hook_fn = partial(patch_head_input, head_index=head, clean_cache=clean_cache)
    hook_fn = partial(
            patch_head_input,
            patched_cache=clean_cache,#clean cache when passed in is the perturbed cache
            head_list=heads_to_patch_list)
    patched_outputs = model.run_with_hooks(
        corrupted_tokens["input_ids"],
        one_zero_attention_mask=corrupted_tokens["attention_mask"],
        token_type_ids = corrupted_tokens['token_type_ids'],
        return_type="embeddings",
        fwd_hooks = [(hook_names_filter, hook_fn)],
    )
    logit_diff = patching_metric(patched_outputs,pooler_layer,dropout_layer,classifier_layer)
    return logit_diff


"""get_activations function: given a list of names, just return the activations values 
    as a dictionary with keys = names requested without patching
    Example use case: 
    name = utils.get_act_name('pattern',layer)
    act = get_activations(model, data.toks,name)[:,head,:,:]
"""
def get_activations(
    model,
    toks,#: Int[Tensor, "batch seq"],

    names#: Union[str, List[str]]
):# -> Union[t.Tensor, ActivationCache]:
    '''
    Uses hooks to return activations from the model.

    If names is a string, returns the activations for that hook name.
    If names is a list of strings, returns the cache containing only those activations.
    '''
    model.reset_hooks()
    names_list = [names] if isinstance(names, str) else names
    _, cache = model.run_with_cache(
        toks["input_ids"],
        one_zero_attention_mask=toks["attention_mask"],
        token_type_ids = toks['token_type_ids'],
        return_type=None,
        names_filter=lambda name: name in names_list,
    )

    return cache[names] if isinstance(names, str) else cache

"""This is activation patching for experiment_IDF: specifically, we swap the highest idf word to the front of the sentence and observe how
this perturbation affects a particular head (receiver's heads) attention pattern on the front token. We hypothesize that these heads attend 
to high idf words and the attention score on the front word will increase if the front word has high idf"""
def get_act_patch_block_every_for_receiver_attention_score(
    model: HookedEncoder, 
    device,
    corrupted_tokens: Float[torch.Tensor, "batch pos"], 
    clean_cache: ActivationCache, 
    patching_metric: Callable[[Float[torch.Tensor, "batch pos d_vocab"]], float],
    receiver_heads,
    first_word_position,
    components_to_patch = ["resid_pre", "attn_out", "mlp_out"]
    #receiver_input,
) -> Float[torch.Tensor, "3 layer pos"]:
    print("first_word_position",first_word_position)
    model.reset_hooks()
    _, seq_len = corrupted_tokens["input_ids"].size()
    results = torch.zeros(3, model.cfg.n_layers, seq_len, device=device, dtype=torch.float32)
    receiver_layers = set(next(zip(*receiver_heads)))
    receiver_hook_names = [utils.get_act_name('pattern', layer) for layer in receiver_layers]
    receiver_hook_names_filter = lambda name: name in receiver_hook_names
    len_first_word = first_word_position[1]-first_word_position[0]+1 
    # send tokens to device if not already there
    corrupted_tokens["input_ids"] = corrupted_tokens["input_ids"].to(device)
    corrupted_tokens["attention_mask"] = corrupted_tokens["attention_mask"].to(device)

    #for component_idx, component in enumerate(["resid_pre", "attn_out", "mlp_out"]):
    for component_idx, component in enumerate(components_to_patch):
        print("Patching:", component)
        #for layer in tqdm(range(model.cfg.n_layers)):
        for layer in tqdm(range(max(receiver_layers))):
            total_score = 0
            score_per_position = []
            patch_name_filter = lambda name: name==utils.get_act_name(component, layer)
            for position in range(seq_len):
                hook_fn = partial( #partial() means you are updating the 3rd-last parameters of the hook function to create some variations
                    patch_residual_component, #hook_fn name
                    pos=position, 
                    clean_cache=clean_cache
                )
                model.add_hook(patch_name_filter, hook_fn,level=1)
                _, patched_cache = model.run_with_cache(
                        corrupted_tokens["input_ids"],
                        one_zero_attention_mask=corrupted_tokens["attention_mask"],
                        token_type_ids = corrupted_tokens['token_type_ids'],
                        names_filter=receiver_hook_names_filter,
                        return_type=None
                    )
                for index, (receiver_l,receiver_h) in enumerate(receiver_heads):
                    if receiver_l==10:
                        #patched_cache[utils.get_act_name('pattern', receiver_l)][0,receiver_h,0,:]: this is cls token attention on all 
                        cls_attention_on_Ks = patched_cache[utils.get_act_name('pattern', receiver_l)][0,receiver_h,0,:]#[seqK]
                        #print(cls_attention_on_Ks)
                        patched_attention_score = cls_attention_on_Ks[first_word_position[0]:first_word_position[1]+1].sum()/len_first_word
                    else:
                        all_tokens_attention_on_Ks = patched_cache[utils.get_act_name('pattern', receiver_l)][0,receiver_h,:,:].sum(dim=0)#batch,layer,seqQ,seqK
                        patched_attention_score = all_tokens_attention_on_Ks[first_word_position[0]:first_word_position[1]+1].sum()/len_first_word
                    total_score+=patched_attention_score
                score_per_position.append(patched_attention_score)
                results[component_idx, layer, position] = patching_metric(patched_attention_score)
           
            #print("layer",layer, score_per_position)
            print("layer",layer, [s.item() for s in score_per_position][:20])
            #if layer==7:
            #    print("layer 7 score",score_per_position)
            #if layer==8:
            #    print("layer 8 score",score_per_position)
    return results


def get_act_patch_attn_head_out_all_pos_for_receiver_attention_score( #TODO: v is not attn_head_out right?
    model: HookedEncoder, 
    device,
    corrupted_tokens: Float[torch.Tensor, "batch pos"], 
    clean_cache: ActivationCache, 
    patching_metric: Callable,
    receiver_heads,
    first_word_position,
) -> Float[torch.Tensor, "layer head"]:

    model.reset_hooks()
    results = torch.zeros(model.cfg.n_layers, model.cfg.n_heads, device=device, dtype=torch.float32)
    receiver_layers = set(next(zip(*receiver_heads)))
    receiver_hook_names = [utils.get_act_name('pattern', layer) for layer in receiver_layers]
    receiver_hook_names_filter = lambda name: name in receiver_hook_names
    len_first_word = first_word_position[1]-first_word_position[0]+1 
    print("Patching: attn_heads")
    for layer in tqdm(range(model.cfg.n_layers)):
        for head in range(model.cfg.n_heads):
            #total_score=0
            hook_fn = partial(patch_head_vector, head_index=head, clean_cache=clean_cache)
            patch_name_filter = lambda name: name==utils.get_act_name("v", layer)
            model.add_hook(patch_name_filter, hook_fn,level=1)
            _, patched_cache = model.run_with_cache(
                    corrupted_tokens["input_ids"],
                    one_zero_attention_mask=corrupted_tokens["attention_mask"],
                    token_type_ids = corrupted_tokens['token_type_ids'],
                    names_filter=receiver_hook_names_filter,
                    return_type=None
                )
            for index, (receiver_l,receiver_h) in enumerate(receiver_heads):
                if receiver_l==10:
                    #patched_cache[utils.get_act_name('pattern', receiver_l)][0,receiver_h,0,:]: this is cls token attention on all 
                    cls_attention_on_Ks = patched_cache[utils.get_act_name('pattern', receiver_l)][0,receiver_h,0,:]#[seqK]
                    #print(cls_attention_on_Ks)
                    patched_attention_score = cls_attention_on_Ks[first_word_position[0]:first_word_position[1]+1].sum()/len_first_word
                else:
                    all_tokens_attention_on_Ks = patched_cache[utils.get_act_name('pattern', receiver_l)][0,receiver_h,:,:].sum(dim=0)#batch,layer,seqQ,seqK
                    patched_attention_score = all_tokens_attention_on_Ks[first_word_position[0]:first_word_position[1]+1].sum()/len_first_word
                #total_score+=patched_attention_score
            results[layer, head] = patching_metric(patched_attention_score)
    return results


def head_ablation_hook(
    value: Float[torch.Tensor, "batch pos head_index d_head"],
    hook,
    heads_to_ablate,
) -> Float[torch.Tensor, "batch pos head_index d_head"]:
    
    #print(hook.name)
    # print(value.shape)

    hook_layer = int(hook.name.split(".")[1])
    for layer, head in heads_to_ablate:
        if layer == hook_layer:
            value[:, :, head] = 0
            #print(value[:, :, head])

    return value

def head_mean_ablation_hook(
    value: Float[torch.Tensor, "batch pos head_index d_head"],
    hook,
    heads_to_ablate,
) -> Float[torch.Tensor, "batch pos head_index d_head"]:
    
    #print(hook.name)
    # print(value.shape)

    hook_layer = int(hook.name.split(".")[1])
    for layer, head in heads_to_ablate:
        if layer == hook_layer:
            value[:, :, head] = 0
            #print(value[:, :, head])

    return value

def attn_out_ablation_hook(
    value: Float[torch.Tensor, "batch seq d_model"],
    hook,
    layers,
) -> Float[torch.Tensor, "batch seq d_model"]:
    
    print(hook.name)
    # print(value.shape)

    hook_layer = int(hook.name.split(".")[1])
    for layer in layers:
        if layer == hook_layer:
            value = 0


    return value

