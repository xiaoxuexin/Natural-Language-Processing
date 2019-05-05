# Regular expression

1  Background

Email  addresses  are  everywhere  online.   Especially  in  personal  web  pages,  people  providetheir email as way of easily getting in touch with them.  However, unscrupulous spammersalso look for these addresses to send unwanted email to people.  As a result, some web pageauthors  have  resorted  to  obfuscating  their  address  so  that  a  human  could  still  figure  outwhat the address is without a machine being able to easily detect it.  For example, someonemight writemyname@domain.eduasmyname at domain dot edu.

2  Task

You’ve been asked to perform a security audit for a large university. They want to know whatkinds of email addresses might be recoverable from each web page.  Conveniently,  they’vealready put together all of the web pages for you into a single file, where the HTML pagefor each page is on one line.  Further, every page is guaranteed to have one email on it atmost, since no one lists two email addresses for themselves on a page.  However, not everyonelists their email on a page, so some pages have no email addresses!  The big challenge is thatthere is no consistency in how the addresses are formatted!
