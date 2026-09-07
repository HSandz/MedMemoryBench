from copy import deepcopy

from methods.smart_mem0.read_semantic_closure import ReadSemanticClosureMixin

Mixin = ReadSemanticClosureMixin


class Base:
    def __init__(self):
        self._active_controller_seeds = []
        self._last_option_probe_coverage = {}
        self._last_proposition_probe_coverage = {}
        self._last_candidate_local_coverage = {}
        self._last_candidate_shared_context_ids = []

    @staticmethod
    def _rc_text(value):
        return ' '.join(str(value or '').casefold().replace('_', ' ').split())

    @staticmethod
    def _snapshot(value):
        return deepcopy(value)

    @staticmethod
    def _memory_value(memory):
        return memory.get('value', '')

    def _rc_normalize_ir(self, parsed, question, frame):
        del question, frame
        return deepcopy(parsed)

    def _controller_plan(self, ir, question, frame):
        del ir, question, frame
        return {
            'budget_tier': 'SMALL',
            'required_slots': [{
                'id': 'r1', 'type': 'TEMPORAL', 'time_axis': 'event_time',
                'time_relation': 'EARLIEST', 'selector': {'axis':'event_time','relation':'EARLIEST'},
            }],
            'semantic_relations': [{'type':'VERIFY_SOURCE','from':'r1','to':''}],
            'operations': [
                {'op':'SEARCH_FAMILY','query':'metformin 1500','produces':['r1'],'family_mode':'anchor'}
            ],
            'retrieval_budget_basis': {'evidence_obligations': 1, 'physical_operations': 1},
        }

    def _make_deterministic_recovery_plan(self, missing_slots, question, existing_plan):
        del missing_slots, question
        return deepcopy(existing_plan)

    def _semantic_operation_search(self, query, top_k, strategy, frame=None, option_queries=None):
        del query, frame, option_queries
        self._last_option_probe_coverage = {'A': ['noise'], 'B': []}
        self._last_proposition_probe_coverage = deepcopy(self._last_option_probe_coverage)
        return [{'id':'noise','claim':'unrelated'}][:top_k]

    def prepare_batch_query(self, question, system_message=None, **kwargs):
        del question, system_message, kwargs
        return {
            'messages':[{'role':'system','content':'base'}],
            'extra':{
                'candidate_set': {'candidates': {'A':'Cefuroxime','B':'Amoxicillin'}},
                'plan': {'reasoning_bridges':[{'type':'INFER'}]},
            },
        }


class Harness(Mixin, Base):
    pass


def test_source_verify_exact_non_date_uses_document_axis():
    h = Harness()
    ir = {
        'answer_type':'TEXT',
        'requirements':[{
            'id':'r1',
            'time_constraint': {'axis':'event_time','relation':'EXACT','anchor':'2024-03-01','end':''},
            'selector': {'axis':'event_time','relation':'EXACT','anchor':'2024-03-01','end':''},
        }],
        'relations':[{'type':'VERIFY_SOURCE','from':'r1','to':''}],
        'normalization_actions':[],
    }
    out = h._rc_normalize_ir(ir, 'q', None)
    assert out['requirements'][0]['selector']['axis'] == 'document_time'
    assert out['requirements'][0]['time_constraint']['axis'] == 'document_time'
    assert out['normalization_actions'][-1]['action'] == 'SOURCE_FILTER_AXIS_NORMALIZED'


def test_date_projection_is_not_rewritten_to_document_axis():
    h = Harness()
    ir = {
        'answer_type':'DATE',
        'requirements':[{'id':'r1','time_constraint':{'axis':'event_time','relation':'EXACT','anchor':'2024-03-01'}}],
        'relations':[{'type':'VERIFY_SOURCE','from':'r1','to':''}],
    }
    out = h._rc_normalize_ir(ir, 'q', None)
    assert out['requirements'][0]['time_constraint']['axis'] == 'event_time'


def test_small_extremum_keeps_search_and_free_select_and_verify():
    h = Harness()
    plan = h._controller_plan({}, 'q', None)
    assert plan['budget_tier'] == 'SMALL'
    assert [op['op'] for op in plan['operations']] == ['SEARCH_FAMILY','SELECT','VERIFY_SOURCE']
    assert plan['operations'][0]['family_mode'] == 'temporal_extremum'
    basis = plan['retrieval_budget_basis']
    assert basis['logical_work_operations'] == 1
    assert basis['deterministic_transforms'] == 2


def test_candidate_exact_seed_is_preserved_as_local_recall():
    h = Harness()
    h._active_controller_seeds = [
        {'id':'m96','claim':'Patient is allergic to cefuroxime (cephalosporins).','object_anchor':'cefuroxime','entities':['cefuroxime']}
    ]
    result = h._semantic_operation_search(
        'Which antibiotic is safe?', 4, 'SHARED_OPTIONS',
        option_queries=[{'label':'A','query':'Cefuroxime'},{'label':'B','query':'Amoxicillin'}],
    )
    assert result[0]['id'] == 'm96'
    assert h._last_option_probe_coverage['A'][0] == 'm96'
    assert 'm96' in h._last_candidate_local_coverage['A']
    assert 'm96' not in h._last_candidate_shared_context_ids


def test_closure_prompt_is_compact_and_mode_neutral():
    h = Harness()
    prepared = h.prepare_batch_query('q')
    text = prepared['messages'][0]['content']
    assert 'Candidate closure:' in text
    assert 'Reasoning closure:' in text
    assert 'EFF' not in text and 'MIX' not in text
    assert prepared['extra']['semantic_closure_materialized'] is True
