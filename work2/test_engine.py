"""Scientific contract checks; no published-performance assertions."""
import unittest
from dataclasses import replace
import numpy as np
from engine import Config, ATTACKS, neighbors, raw_payoffs, simulate, local_controller, reported_message

class EngineContracts(unittest.TestCase):
    def test_neighborhood_geometry(self):
        for mode,count in [(1,4),(2,12)]:
            nb=neighbors(5,mode)
            self.assertEqual(nb.shape,(25,count))
            for i,row in enumerate(nb):
                self.assertNotIn(i,row)
                self.assertEqual(len(set(row)),count)

    def test_payoff_against_direct_group_accounting(self):
        L=5; nb=neighbors(L,1); rng=np.random.default_rng(17)
        for r in [1.,3.6,5.]:
            for _ in range(20):
                a=rng.integers(0,2,L*L)
                brute=np.zeros(L*L)
                for center in range(L*L):
                    group=[center,*nb[center]]
                    nc=sum(a[i]==0 for i in group)
                    for i in group:
                        brute[i]+=r*nc/5-(a[i]==0)
                actual=raw_payoffs(a,nb,r)
                np.testing.assert_allclose(actual,brute,atol=1e-13)
                self.assertAlmostEqual(actual.sum(),5*(r-1)*np.sum(a==0),places=11)

    def test_extreme_payoffs(self):
        nb=neighbors(5,1);r=3.6
        np.testing.assert_allclose(raw_payoffs(np.ones(25,dtype=int),nb,r),0)
        np.testing.assert_allclose(raw_payoffs(np.zeros(25,dtype=int),nb,r),5*(r-1))
        a=np.ones(25,dtype=int);a[12]=0
        self.assertAlmostEqual(raw_payoffs(a,nb,r)[12],r-5)
        a=1-a
        self.assertAlmostEqual(raw_payoffs(a,nb,r)[12],4*r)

    def test_reward_belongs_to_executed_action(self):
        cfg=Config(L=5,steps=1,record_every=1,method='iql',alpha=1,gamma=0,payoff_weight=0.7)
        run=simulate(cfg);a=run['actions']
        p=raw_payoffs(a,neighbors(5,1),cfg.r)
        expected=.7*(p-cfg.r+5)/(3*cfg.r+5)+.3*(1-a)
        # Initial reputation implies state zero for every node.
        np.testing.assert_allclose(run['q'][np.arange(25),0,a],expected,atol=1e-14)

    def test_zero_kappa_is_same_random_path_as_iql(self):
        cfg=Config(L=5,steps=80,record_every=20,method='iql',rho=.2,attack='high_defect')
        reference=simulate(cfg)
        for method in ['ni_global','ni_local','trimmed_local']:
            other=simulate(replace(cfg,method=method,kappa=0))
            np.testing.assert_array_equal(reference['q'],other['q'])
            np.testing.assert_array_equal(reference['trajectory'],other['trajectory'])

    def test_messages_do_not_change_iql(self):
        for state in ['reputation','own_action']:
            cfg=Config(L=5,steps=80,record_every=20,method='iql',state_mode=state)
            clean=simulate(cfg)
            for attack in ATTACKS:
                attacked=simulate(replace(cfg,rho=.2,attack=attack))
                np.testing.assert_array_equal(clean['q'],attacked['q'])
                np.testing.assert_array_equal(clean['trajectory'],attacked['trajectory'])

    def test_no_faults_and_repeatability(self):
        cfg=Config(L=5,steps=80,record_every=20)
        first=simulate(cfg);second=simulate(cfg)
        np.testing.assert_array_equal(first['q'],second['q'])
        np.testing.assert_array_equal(first['trajectory'],second['trajectory'])
        for attack in ATTACKS:
            no_fault_attack=simulate(replace(cfg,attack=attack))
            np.testing.assert_array_equal(first['q'],no_fault_attack['q'])
            np.testing.assert_array_equal(first['trajectory'],no_fault_attack['trajectory'])
        self.assertEqual(first['tail']['selected_bad_fraction'],0)

    def test_message_transformations_and_burst_boundaries(self):
        rng=np.random.default_rng(7)
        cases={'none':(0,.4),'high_defect':(1,1.),'high_only':(0,1.),
               'flip_action':(1,.4),'moderate_defect':(1,.6),'high_cooperate':(0,1.)}
        for attack,expected in cases.items():
            actual=reported_message(0,.4,ATTACKS[attack],0,rng)
            self.assertEqual(actual[0],expected[0])
            self.assertAlmostEqual(actual[1],expected[1])
        self.assertEqual(reported_message(1,.4,ATTACKS['flip_action'],0,rng),(0,.4))
        self.assertEqual(reported_message(0,.9,ATTACKS['moderate_defect'],0,rng),(1,1.))
        for step in [0,499,1000,1499]:
            self.assertEqual(reported_message(0,.4,ATTACKS['burst_defect'],step,rng),(1,1.))
        for step in [500,999,1500,1999]:
            self.assertEqual(reported_message(0,.4,ATTACKS['burst_defect'],step,rng),(0,.4))

    def test_random_messages_match_independent_generator(self):
        rng=np.random.default_rng(17)
        expected=np.random.default_rng(17).random((100,2))
        for step,(reward,action_draw) in enumerate(expected):
            action,value=reported_message(step%2,.4,ATTACKS['random_message'],step,rng)
            self.assertEqual(action,int(action_draw>=.5))
            self.assertEqual(value,reward)
        cfg=Config(L=5,steps=80,record_every=20,rho=.2,attack='random_message')
        first=simulate(cfg);second=simulate(cfg)
        np.testing.assert_array_equal(first['q'],second['q'])
        np.testing.assert_array_equal(first['trajectory'],second['trajectory'])

    def test_own_action_state_uses_current_action_as_next_state(self):
        cfg=Config(L=5,steps=1,record_every=1,method='iql',state_mode='own_action',
                   alpha=1.,gamma=.3,payoff_weight=.7,epsilon_initial=1.,epsilon_min=1.)
        run=simulate(cfg);actions=run['actions']
        initial=np.random.default_rng(np.random.SeedSequence(cfg.seed).spawn(1)[0]).uniform(-.01,.01,(25,2,2))
        payoff=raw_payoffs(actions,neighbors(5,1),cfg.r)
        reward=.7*(payoff-cfg.r+5)/(3*cfg.r+5)+.3*(1-actions)
        expected=initial.copy()
        for i,action in enumerate(actions):
            expected[i,1,action]=reward[i]+.3*max(initial[i,action])
        self.assertEqual(set(actions),{0,1})
        np.testing.assert_allclose(run['q'],expected,rtol=0,atol=1e-14)

    def test_own_action_iql_does_not_use_perception_radius(self):
        cfg=Config(L=5,steps=120,record_every=20,method='iql',state_mode='own_action',M=1)
        first=simulate(cfg);second=simulate(replace(cfg,M=2))
        np.testing.assert_array_equal(first['q'],second['q'])
        np.testing.assert_array_equal(first['trajectory'],second['trajectory'])

    def test_fixed_sender_count_and_welfare_trajectory(self):
        cfg=Config(L=5,steps=80,record_every=20,rho=.2,attack='high_defect')
        run=simulate(cfg)
        self.assertEqual(run['bad_mask'].sum(),5)
        self.assertAlmostEqual(run['candidate_bad_fraction'],.2)
        np.testing.assert_allclose(run['trajectory'][:,2],
                                   5*(cfg.r-1)*run['trajectory'][:,1],atol=1e-12)
        self.assertTrue(np.isfinite(run['q']).all())
        self.assertLessEqual(run['tail']['mean_abs_ni'],cfg.kappa)

    def test_controller_selection_matches_manual_scores(self):
        a=np.array([0,1,0,1]);r=np.array([.2,.9,.8,.4]);pred=np.array([.5,.6])
        theta=np.array([1.,.2,-.8,-.4,.6,.3,.1,-.7,-.2,.8,.4,-.5])
        x=np.array([[value-.3,1 if action==0 else -1,abs(value-.6),abs(value-pred[action]),1-action]
                    for value,action in zip(r,a)])
        index,gate=local_controller(0,.3,a,r,pred,.7,theta,5,.25,1.)
        expected=int(np.argmax(x@theta[:5]))
        self.assertEqual(index,expected)
        logit=theta[11]+theta[10]/1.7+x[expected]@theta[5:10]
        self.assertAlmostEqual(gate,1/(1+np.exp(-logit)))

    def test_cooperation_prior_and_uniform_ties(self):
        a=np.array([1,0,0,1]);r=np.array([1.,.4,.4,.8])
        for draw,expected in [(.1,1),(.9,2)]:
            index,gate=local_controller(1,.5,a,r,np.empty(0),0.,np.zeros(12),4,draw,1.)
            self.assertEqual(index,expected);self.assertEqual(gate,1.)

    def test_zero_gate_is_exact_iql(self):
        cfg=Config(L=5,steps=100,record_every=20,method='iql',rho=.2,attack='high_defect')
        reference=simulate(cfg)
        for method in ['learned','gate_only','selection_only']:
            run=simulate(replace(cfg,method=method,controller=(1.,0.,-1.,-1.,2.,0.,0.,0.,0.,0.,0.,0.),gate_scale=0))
            np.testing.assert_array_equal(reference['q'],run['q'])
            np.testing.assert_array_equal(reference['trajectory'],run['trajectory'])

    def test_learned_update_bound_and_reproducibility(self):
        cfg=Config(L=5,steps=100,record_every=20,method='learned',kappa=.5,
                   controller=(1.,.2,-1.,-.5,2.,0.,0.,-2.,-1.,1.,.2,-.1),rho=.2,attack='high_defect')
        run=simulate(cfg);repeat=simulate(cfg)
        np.testing.assert_array_equal(run['q'],repeat['q'])
        self.assertGreaterEqual(run['tail']['mean_gate'],0)
        self.assertLessEqual(run['tail']['mean_gate'],1)
        self.assertLessEqual(run['tail']['mean_abs_ni'],cfg.kappa*run['tail']['mean_gate']+1e-12)

    def test_semantic_scoring_can_reproduce_cooperation_priority(self):
        cfg=Config(L=5,steps=200,record_every=20,method='cooperation_first',rho=.2,attack='high_defect')
        theta=(1.,0.,0.,0.,2.,0.,0.,0.,0.,0.,0.,0.)
        reference=simulate(cfg)
        other=simulate(replace(cfg,method='selection_only',controller=theta))
        np.testing.assert_array_equal(reference['q'],other['q'])
        np.testing.assert_array_equal(reference['trajectory'],other['trajectory'])

    def test_cooperation_only_rejects_all_defection_reports(self):
        cfg=Config(L=5,steps=100,record_every=20,method='cooperation_only',rho=1.,attack='high_defect')
        run=simulate(cfg)
        iql=simulate(replace(cfg,method='iql'))
        np.testing.assert_array_equal(run['q'],iql['q'])
        self.assertEqual(run['tail']['mean_gate'],0.)
        self.assertEqual(run['tail']['active_ni_fraction'],0.)
        self.assertEqual(run['tail']['mean_abs_ni'],0.)
        self.assertIsNone(run['tail']['active_selected_bad_fraction'])

    def test_cooperation_only_has_nonnegative_single_step_preference_change(self):
        cfg=Config(L=5,steps=1,record_every=1,method='cooperation_only',rho=.2,attack='high_defect')
        run=simulate(cfg);iql=simulate(replace(cfg,method='iql'))
        delta=(run['q'][:,0,0]-run['q'][:,0,1])-(iql['q'][:,0,0]-iql['q'][:,0,1])
        self.assertTrue((delta>=-1e-14).all())
        self.assertGreater(delta.max(),0.)

if __name__=='__main__': unittest.main()
