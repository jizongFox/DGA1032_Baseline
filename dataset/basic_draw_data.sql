select basetable.patient_number,
       basetable.scan_number,
       basetable.slice_number,
       round(cast(basetable.slice_number as float) / cast(basetable.max as float) * 10, 2) as 'equvalent_slice_number',
       basetable.class1
from (
     select sizeinfo.patient_number,
             scan_number,
             cast(slice_number as decimal)               as 'slice_number',
             class1,
             cast(max_slices_per_patient.max as decimal) as `max`
      from sizeinfo
             left join (select patient_number, max(slice_number) as max
                        from sizeinfo
                        group by patient_number
                        order by patient_number) as max_slices_per_patient
               on sizeinfo.patient_number = max_slices_per_patient.patient_number
      order by sizeinfo.patient_number, sizeinfo.scan_number, slice_number) as basetable
where basetable.scan_number == '01'
order by basetable.patient_number, basetable.scan_number, basetable.slice_number;






select cast (t.equvalent_slice_number as integer) as equvalent_slice_number, avg(t.class1) from (

select basetable.patient_number,
       basetable.scan_number,
       basetable.slice_number,
       round(cast(basetable.slice_number as float) / cast(basetable.max as float) * 10, 2) as 'equvalent_slice_number',
       basetable.class1
from (
     select sizeinfo.patient_number,
             scan_number,
             cast(slice_number as decimal)               as 'slice_number',
             class1,
             cast(max_slices_per_patient.max as decimal) as `max`
      from sizeinfo
             left join (select patient_number, max(slice_number) as max
                        from sizeinfo
                        group by patient_number
                        order by patient_number) as max_slices_per_patient
               on sizeinfo.patient_number = max_slices_per_patient.patient_number
      order by sizeinfo.patient_number, sizeinfo.scan_number, slice_number) as basetable
where basetable.scan_number == '01'
order by basetable.patient_number, basetable.scan_number, basetable.slice_number ) as t
group by cast (t.equvalent_slice_number as integer)

