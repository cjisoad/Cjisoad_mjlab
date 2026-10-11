"""Bounded PPO chunks; validation decides progression, sealed tests decide acceptance."""
from dataclasses import asdict
from pathlib import Path
import json,math
import torch
from .gates import GateConfig,qualifies
from .data import verify_assets
from .evaluation import evaluate_policy

def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def train(args):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends
    from src.tasks.transition_down_t.data import verify_teachers,file_sha256
    from .env_cfg import landing_env_cfg
    from .rl import LandingRunner,landing_ppo_cfg
    from .safety import install_monitor
    output=Path(args.output)
    if output.exists() and any(output.iterdir()):raise ValueError('Use a fresh output directory, including for resumed runs')
    output.mkdir(parents=True,exist_ok=True)
    gate=GateConfig(**json.loads(Path(args.gate_config).read_text())) if args.gate_config else GateConfig()
    assets=verify_assets();verify_teachers(args.quad_teacher,args.biped_teacher)
    configure_torch_backends();torch.set_num_threads(1);torch.manual_seed(args.seed)
    cfg=landing_env_cfg();cfg.scene.num_envs=args.num_envs;cfg.seed=args.seed
    env=ManagerBasedRlEnv(cfg,device=args.device);runner=None
    try:
        monitor=install_monitor(env,gate);wrapped=RslRlVecEnvWrapper(env,clip_actions=10.)
        agent=landing_ppo_cfg();agent.seed=args.seed;agent.save_interval=100
        if args.smoke:
            agent.num_steps_per_env=8;agent.algorithm.num_learning_epochs=1;agent.algorithm.num_mini_batches=2
        runner=LandingRunner(wrapped,asdict(agent),str(output),device=args.device)
        if args.resume:runner.resume(args.resume)
        write_json(output/'run.json',dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            gate=gate.to_dict(),assets=assets,initialization='strict_resume' if args.resume else 'from_scratch',
            optimization_continuous=True,torch=torch.__version__,
            gpu=torch.cuda.get_device_name(torch.device(args.device)) if args.device.startswith('cuda') else None))
        completed=runner.completed_updates
        if completed>=args.max_updates:raise ValueError('Budget is already exhausted by resumed checkpoint')
        budget=min(args.max_updates,2) if args.smoke else args.max_updates
        before=[p.detach().clone() for p in runner.alg.actor.parameters()] if args.smoke else None
        while completed<budget and runner.term.course.phase!='acceptance':
            count=min(args.eval_interval,budget-completed)
            runner.learn(count);completed=runner.completed_updates
            if not all(torch.isfinite(p).all() for p in runner.alg.actor.parameters()):raise ValueError('Nonfinite actor after PPO')
            if runner.logger.writer is not None:
                runner.logger.writer.flush();runner.logger.writer.close();runner.logger.writer=None
            stage=runner.term.course.to_dict();reports={}
            def evaluate(name,count,**kwargs):
                report=evaluate_policy(runner.alg.actor,gate=gate,device=args.device,group=name,count=count,
                    seed=910000+completed*10+len(reports),**kwargs)
                write_json(output/f'evaluation_{completed:06d}_{name}.json',report)
                reports[name]=report
            evaluate('nominal',2 if args.smoke else gate.nominal_attempts)
            if stage['phase']=='nominal':
                for name in ('push_forward','push_backward'):evaluate(name,2 if args.smoke else gate.probe_attempts)
            else:evaluate('entry',2 if args.smoke else gate.entry_attempts,level=stage['level'],split='validation')
            promoted=runner.term.course.observe(reports,gate) if not args.smoke else False
            event=dict(updates_completed=completed,before=stage,course=runner.term.course.to_dict(),promoted=promoted,
                reports={k:{a:v for a,v in r.items() if a!='records'} for k,r in reports.items()})
            with (output/'course.jsonl').open('a') as stream:stream.write(json.dumps(event,allow_nan=False)+'\n')
            print('LANDING_COURSE '+json.dumps(event,allow_nan=False),flush=True)
            runner.save(str(output/'latest.pt'))
            if promoted:runner.save(str(output/f'passed_{stage["phase"]}_level{stage["level"]}.pt'))
        runner.save(str(output/'latest.pt'))
        status='smoke_completed' if args.smoke else 'budget_exhausted'
        accepted=False;final={}
        if args.smoke:
            assert completed>0 and any(not torch.equal(a,b) for a,b in zip(before,runner.alg.actor.parameters()))
            reports['entry']=evaluate_policy(runner.alg.actor,gate=gate,device=args.device,group='entry',count=2,
                level=0,split='validation',seed=920001)
            write_json(output/'smoke_entry.json',reports['entry'])
            from .chain import evaluate_chain
            chain=evaluate_chain(runner.alg.actor,gate=gate,device=args.device,count=4,seed=920002,
                quad_teacher=args.quad_teacher,biped_teacher=args.biped_teacher)
            write_json(output/'smoke_chain.json',chain)
            if not any(h['kind']=='BIPED_DOWN' for h in chain['handoffs']):raise AssertionError('No source teacher reached handoff')
            saved=output/'latest.pt'
            runner.resume(saved)
            if runner.completed_updates!=completed:raise AssertionError('Resume clock differs')
            write_json(output/'verification.json',dict(smoke_only=True,actor_updated=True,strict_resume=True,
                physical_substep_monitor=True,entry_evaluation=True,live_teacher_handoff=True,physics_dt=env.physics_dt,updates_completed=completed,
                checkpoint=str(saved.resolve()),never_accepts_policy=True))
        elif runner.term.course.phase=='acceptance':
            from .chain import evaluate_chain
            runner.save(str(output/'candidate.pt'))
            for group in ('nominal','base','joints','combined'):
                report=evaluate_policy(runner.alg.actor,gate=gate,device=args.device,group=group,count=gate.nominal_attempts,seed=1800000+len(final))
                final[group]=report;write_json(output/f'final_{group}.json',report)
            final['augmented']=evaluate_policy(runner.alg.actor,gate=gate,device=args.device,group='combined',count=gate.nominal_attempts,seed=1800100,augmented=True)
            write_json(output/'final_augmented.json',final['augmented'])
            for level in range(4):
                name=f'entry_level{level}'
                final[name]=evaluate_policy(runner.alg.actor,gate=gate,device=args.device,group='entry',count=gate.entry_attempts,level=level,split='final',seed=1900000+level)
                write_json(output/f'final_{name}.json',final[name])
            final['chain']=evaluate_chain(runner.alg.actor,gate=gate,device=args.device,count=128,seed=2000000,
                quad_teacher=args.quad_teacher,biped_teacher=args.biped_teacher)
            write_json(output/'final_chain.json',final['chain'])
            accepted=all(qualifies(r,gate.final_nominal_rate if name=='nominal' else gate.final_rate,
                gate.entry_attempts if name.startswith('entry') else 128 if name=='chain' else gate.nominal_attempts,
                distinct=name.startswith('entry')) for name,r in final.items())
            status='accepted' if accepted else 'acceptance_failed'
            if accepted:runner.save(str(output/'accepted.pt'))
        result=dict(status=status,accepted=accepted,updates_completed=completed,course=runner.term.course.to_dict(),
            checkpoint=str((output/('accepted.pt' if accepted else 'latest.pt')).resolve()),
            checkpoint_sha256=file_sha256(output/('accepted.pt' if accepted else 'latest.pt')),
            final_reports={k:{a:v for a,v in r.items() if a not in ('records','handoffs')} for k,r in final.items()})
        write_json(output/'finished.json',result);print('LANDING_FINISHED '+json.dumps(result),flush=True)
        return result
    finally:
        if runner is not None and getattr(runner.logger,'writer',None) is not None:runner.logger.writer.close()
        env.close()
