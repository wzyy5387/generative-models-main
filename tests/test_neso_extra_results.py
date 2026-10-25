import numpy as np
import pytest
import torch
from GEFcom2014.forecast_quality.neso_extra_results import circular_indices
from GEFcom2014.models.neso_extra_baselines import ConditionalVAE


def test_bootstrap_preserves_segment_weights_and_does_not_cross_windows_or_gaps():
    dates=['2023-01-01','2023-01-02','2023-01-03','2023-01-04','2023-01-09']
    windows=['q1','q1','q2','q2','q2']
    indices=circular_indices(dates,windows,3,100,7)
    assert indices.shape==(100,5)
    assert np.isin(indices[:,:2],[0,1]).all()
    assert np.isin(indices[:,2:4],[2,3]).all()
    assert (indices[:,4]==4).all()
    scores=np.array([1,1,9,9,30])
    np.testing.assert_allclose(scores[indices].mean(axis=1),10)
    with pytest.raises(ValueError):
        circular_indices(dates[::-1],windows,3,100,7)


def test_cvae_logvariance_is_converted_to_standard_deviation():
    torch.manual_seed(42)
    model=ConditionalVAE(1,1,latent_dim=1,hidden_dim=1,layers=0)
    with torch.no_grad():
        for p in model.parameters():p.zero_()
        model.encoder[-1].bias[1]=np.log(4.)
        model.decoder[-1].weight[0,0]=1.
    # q(z)=N(0,4), decoder mean=z and observation variance=1.
    # Expected negative ELBO (without constant) = 4/2 + KL(N(0,4)||N(0,1)).
    actual=float(model.loss(torch.zeros(30000,1),torch.zeros(30000,1)))
    assert abs(actual-(2.+.5*(4.-1.-np.log(4.))))<.06
