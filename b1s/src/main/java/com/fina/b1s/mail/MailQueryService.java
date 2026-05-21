package com.fina.b1s.mail;

import com.fina.b1s.dto.MailMessageVO;

import java.util.List;

public interface MailQueryService {

    List<MailMessageVO> listRecent(int limit);

    MailMessageVO getById(Long id);
}
