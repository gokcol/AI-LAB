"""Verify the mathematical bridges, not merely that the demonstrations render."""
import gzip
import struct
import numpy as np
import pytest
from core import attention as a, digits as d
from core.challenges import learning_rate_run, leakage_run, classification_metrics, QUESTIONS, retrieval_scores


def test_attention_mask_and_causal_block():
    X=np.random.default_rng(3).normal(size=(5,8))
    w=a.block_weights(8,3)
    out,trace=a.transformer_block(X,w,heads=2)
    assert out.shape==X.shape
    assert np.allclose(trace['maps'].sum(axis=-1),1)
    assert np.allclose(np.triu(trace['maps'],1),0)
    changed=X.copy(); changed[-1]+=100
    out2,_=a.transformer_block(changed,w,heads=2)
    assert np.allclose(out[:-1],out2[:-1])
    assert not np.allclose(out[-1],out2[-1])
    with pytest.raises(ValueError): a.multihead(X,w,heads=3)


def test_attention_matches_hand_calculation():
    q=np.zeros((2,2));k=q.copy();v=np.array([[2.,4.],[6.,8.]])
    output, weights=a.attention(q,k,v)
    assert np.allclose(output,[[2,4],[4,6]])
    assert np.allclose(weights,[[1,0],[.5,.5]])


def test_digit_gradients_finite_differences():
    rng=np.random.default_rng(9);X=rng.normal(size=(4,3));y=np.array([0,1,0,1])
    p=d.initialize(3,4,2,seed=3)
    _,prob,g=d.loss_gradients(X,y,p)
    assert np.allclose(prob.sum(axis=1),1)
    eps=1e-5
    for key,values in p.items():
        for ix in np.ndindex(values.shape):
            old=values[ix]
            values[ix]=old+eps;plus=d.loss_gradients(X,y,p)[0]
            values[ix]=old-eps;minus=d.loss_gradients(X,y,p)[0]
            values[ix]=old
            assert g[key][ix]==pytest.approx((plus-minus)/(2*eps),abs=1e-6)


def test_numpy_torch_same_updates():
    pytest.importorskip('torch')
    rng=np.random.default_rng(2);X=rng.normal(size=(20,3));y=(X[:,0]>0).astype(int)
    p=d.initialize(3,5,2,seed=7)
    n,nh=d.train_numpy(X,y,p,epochs=3,batch=7)
    t,th=d.train_torch(X,y,p,epochs=3,batch=7)
    assert nh==pytest.approx(th,abs=1e-10)
    for key in p: assert np.allclose(n[key],t[key],atol=1e-10)


def test_idx_parser_and_no_implicit_download(tmp_path):
    file=tmp_path/'labels.gz'
    with gzip.open(file,'wb') as f: f.write(struct.pack('>HBBI',0,8,1,3)+bytes([1,2,3]))
    assert d.read_idx(file).tolist()==[1,2,3]
    with pytest.raises(FileNotFoundError): d.mnist(tmp_path)


def test_challenges_have_expected_failure_modes():
    assert learning_rate_run(.05)[-1,1]<learning_rate_run(.05)[0,1]
    assert learning_rate_run(.25)[-1,1]>learning_rate_run(.25)[0,1]
    leaked,honest,_=leakage_run(7)
    assert leaked==1 and honest<.8
    metrics=classification_metrics(np.array([0]*95+[1]*5),np.zeros(100))
    assert metrics['accuracy']==.95 and metrics['recall']==0
    assert QUESTIONS[-1][1] is None
    assert retrieval_scores(QUESTIONS[-1][0]).max()>0  # similarity is not evidence
