# sql配置类
class Sql:
    SQL_TEST = """
        select * from up_common_ods.ods_common_nsh_bhls
        where gameplayid = '51000016-319-8193-0-9-1619787600' 
        order by ts desc
    """

    SQL_GAME_LOG = """
        -- 最终改进版，这里为了合并多天数据集到一个文件夹
        select concat(gameplayid,ds) as gameplayid,battlefield_duration,finish,res1,res2,shiqi1,shiqi2,item1,item2,jidi1,jidi2,liangcang1,liangcang2,players,total_kill,side1_kill,side2_kill,ds from 
            (select * from up_nsh_dwd.dwd_nsh_bhls_win_predict_profile_add_d 
             where ds = '{}'  and gameplayid in( -- 这里声明日期
                select gameplayid from (
                    select * ,row_number() over (partition by gameplayid order by battlefield_duration) as ranking --窗口内排序
                    from up_nsh_dwd.dwd_nsh_bhls_win_predict_profile_add_d
                    where ds = '{}' and gameplayid in
                        (select gameplayid
                            from(
                            select gameplayid,count(*) as duration 
                            from  up_nsh_dwd.dwd_nsh_bhls_win_predict_profile_add_d
                            where ds = '{}' and players !='[]' --去空后长度大于20
                            group by gameplayid
                            ) as a
                        where duration >=129)) as b --抛弃较短的比赛,这里要更大才行
                where ranking=1 and battlefield_duration<1) --排除垃圾数据
            ) as c
        where players !='[]'
        -- order by gameplayid, battlefield_duration 内存暴了
    """