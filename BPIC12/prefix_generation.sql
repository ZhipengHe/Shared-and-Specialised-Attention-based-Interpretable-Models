use BPIC

drop table if exists #cases

select a.caseid, min(a.start_timestamp) start_time, max(a.end_timestamp) end_time
into #cases

from BPIC_2012_log a
group by a.caseid

drop table if exists #traces

select a.*, b.start_time, b.end_time, ROW_NUMBER() OVER (
      PARTITION BY a.caseid
      ORDER BY start_timestamp asc
   ) task_index
into #traces
from BPIC_2012_log a
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
where o.task in ('A_PREACCEPTED','W_Nabellen offertes')
and o.task <> o2.task



drop table if exists #final_prefixes
select cast(o.caseid as varchar)+'_'+cast(m.milestone_id as varchar) as prefix_id, o.caseid, o.task, o.event_type, o.role,  o.start_timestamp, o.end_timestamp,o.start_time as trace_start, datediff(hour,o.start_time,o.start_timestamp) as timelapsed, m.next_activity, m.milestone, m.milestone_id, o.task_index
into #final_prefixes
from #traces o
join #milestones m on o.caseid = m.caseid and o.task_index <= m.milestone_index

select milestone, count(distinct prefix_id) from #final_prefixes
group by milestone

select * from #final_prefixes

