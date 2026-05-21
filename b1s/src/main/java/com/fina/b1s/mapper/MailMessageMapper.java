package com.fina.b1s.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.fina.b1s.entity.MailMessage;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface MailMessageMapper extends BaseMapper<MailMessage> {
}
