package com.fina.b1s.mail;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.fina.b1s.dto.MailAttachmentVO;
import com.fina.b1s.dto.MailMessageVO;
import com.fina.b1s.entity.MailAttachment;
import com.fina.b1s.entity.MailMessage;
import com.fina.b1s.mapper.MailAttachmentMapper;
import com.fina.b1s.mapper.MailMessageMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.BeanUtils;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
@RequiredArgsConstructor
public class MailQueryServiceImpl implements MailQueryService {

    private final MailMessageMapper messageMapper;
    private final MailAttachmentMapper attachmentMapper;

    @Override
    public List<MailMessageVO> listRecent(int limit) {
        int safeLimit = Math.max(1, Math.min(limit, 100));
        List<MailMessage> messages = messageMapper.selectList(
                new LambdaQueryWrapper<MailMessage>()
                        .orderByDesc(MailMessage::getId)
                        .last("LIMIT " + safeLimit)
        );
        return messages.stream().map(this::toVO).toList();
    }

    @Override
    public MailMessageVO getById(Long id) {
        MailMessage message = messageMapper.selectById(id);
        if (message == null) {
            throw new IllegalArgumentException("Mail message not found: " + id);
        }
        return toVO(message);
    }

    private MailMessageVO toVO(MailMessage message) {
        MailMessageVO vo = new MailMessageVO();
        BeanUtils.copyProperties(message, vo);
        List<MailAttachment> attachments = attachmentMapper.selectList(
                new LambdaQueryWrapper<MailAttachment>()
                        .eq(MailAttachment::getMailMessageId, message.getId())
                        .orderByAsc(MailAttachment::getId)
        );
        vo.setAttachments(attachments.stream().map(this::toAttachmentVO).toList());
        return vo;
    }

    private MailAttachmentVO toAttachmentVO(MailAttachment attachment) {
        MailAttachmentVO vo = new MailAttachmentVO();
        BeanUtils.copyProperties(attachment, vo);
        return vo;
    }
}
