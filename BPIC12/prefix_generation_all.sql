use BPIC

drop table if exists #BPIC_raw

select
Case_ID as caseid,
Activity as Task,
Complete_Timestamp as end_timestamp,
isnull(resource,'000') as role
into #BPIC_raw

--from dbo.bpic12_translated_complete
from dbo.bpic12_translated_complete_A
--from dbo.bpic12_translated_complete_O
--from dbo.bpic12_translated_complete_W

where lifecycle_transition = 'COMPLETE' 

drop table if exists #exclude
select distinct caseid into #exclude
from #BPIC_raw
where role is NULL

drop table if exists #cases

select a.caseid, min(a.end_timestamp) start_time
into #cases

from #BPIC_raw a
where a.caseid not in (select* from #exclude)
group by a.caseid


drop table if exists #traces

select a.*, b.start_time, ROW_NUMBER() OVER (
      PARTITION BY a.caseid
      ORDER BY end_timestamp asc
   ) task_index
into #traces
from #BPIC_raw a
join #cases b on a.caseid = b.caseid




drop table if exists #milestones

select distinct o.caseid, o.task_index as milestone_index, o.task as milestone, o2.task as next_activity ,
ROW_NUMBER() OVER (
      PARTITION BY o.caseid
      ORDER BY o.task_index
   ) milestone_id
into #milestones

from #traces o join
#traces o2 on o.task_index = o2.task_index-1 and o.caseid = o2.caseid
--where o.task in ('O_SENT')
--and o.task <> o2.task



drop table if exists #final_prefixes
select cast(o.caseid as varchar)+'_'+cast(m.milestone_id as varchar) as prefix_id, o.caseid, o.task,  'role_'+str(o.role) as role,   o.end_timestamp,o.start_time as trace_start, datediff(hour,o.start_time,o.end_timestamp) as timelapsed, m.next_activity, m.milestone, m.milestone_id, o.task_index
into #final_prefixes
from #traces o
join #milestones m on o.caseid = m.caseid and o.task_index <= m.milestone_index

select milestone, count(distinct prefix_id) from #final_prefixes
group by milestone

select max(prefix_length) from ( select f.prefix_id, max(f.task_index) prefix_length from #final_prefixes  f group by f.prefix_id) a

select f.*, a.prefix_length from #final_prefixes f
join ( select f.prefix_id, max(f.task_index) prefix_length from #final_prefixes  f group by f.prefix_id) a on f.prefix_id = a.prefix_id
join (select top 5000 caseid from  (select distinct caseid from #final_prefixes) ca order by newid()) b on b.caseid = f.caseid

