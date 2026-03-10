import styles from "./TemplateGallery.module.css";
import type { CSSProperties } from "react";
import { domains, templates } from "./templateData";
import { getIllustration } from "./Illustrations";

export default function TemplateGallery() {
  return (
    <section className={styles.root} aria-label="Agent Analysis Templates">
      <div className={styles.header}>
        <div className={styles.title}>Agent Analysis Templates</div>
        <div className={styles.subtitle}>Start from proven analysis workflows designed for agents.</div>
      </div>

      <div className={styles.domains}>
        {domains.map((domain) => {
          const domainTemplates = templates.filter((t) => t.domain === domain.id);
          return (
            <div
              key={domain.id}
              className={styles.domainBlock}
              style={{ "--accent": domain.accent } as CSSProperties}
            >
              <div className={styles.domainHeader}>
                <div className={styles.domainName}>
                  <span className={styles.domainDot} />
                  <span>{domain.name}</span>
                </div>
              </div>

              <div className={styles.cards}>
                {domainTemplates.map((t) => (
                  <div key={t.id} className={styles.card}>
                    <div className={styles.iconWrap}>{getIllustration(t.illustrationKey, domain.accent)}</div>
                    <div className={styles.cardText}>
                      <div className={styles.cardTitle}>{t.title}</div>
                      <div className={styles.cardDesc}>{t.description}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

