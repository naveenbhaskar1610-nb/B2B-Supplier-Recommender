-- Market-health analytics over the synthetic supplier base (SQLite dialect).
-- Each block starts with "-- name: <id>" so the app can load and run them.

-- name: suppliers_by_state
SELECT state,
       COUNT(*)                                   AS suppliers,
       ROUND(AVG(rating), 2)                      AS avg_rating,
       ROUND(100.0 * AVG(gst_verified), 1)        AS pct_gst_verified
FROM suppliers
GROUP BY state
ORDER BY suppliers DESC;

-- name: category_health
SELECT o.category,
       COUNT(DISTINCT s.supplier_id)              AS suppliers,
       COUNT(*)                                   AS offers,
       ROUND(100.0 * AVG(s.response_rate), 1)     AS avg_response_pct,
       ROUND(AVG(s.avg_response_hours), 1)        AS avg_reply_hours,
       ROUND(100.0 * AVG(s.gst_verified), 1)      AS pct_gst_verified
FROM offers o
JOIN suppliers s USING (supplier_id)
GROUP BY o.category
ORDER BY suppliers DESC;

-- name: trust_tiers
SELECT CASE
         WHEN gst_verified = 1 AND trust_badge = 1 THEN '1. GST + trust badge'
         WHEN gst_verified = 1                    THEN '2. GST only'
         ELSE                                          '3. Unverified'
       END                                        AS tier,
       COUNT(*)                                   AS suppliers,
       ROUND(AVG(rating), 2)                      AS avg_rating,
       ROUND(100.0 * AVG(response_rate), 1)       AS avg_response_pct,
       ROUND(AVG(avg_response_hours), 1)          AS avg_reply_hours
FROM suppliers
GROUP BY tier
ORDER BY tier;

-- name: top_responsive_per_category
WITH ranked AS (
  SELECT category, name, city, response_rate, rating,
         ROW_NUMBER() OVER (PARTITION BY category
                            ORDER BY response_rate DESC, rating DESC) AS rn
  FROM suppliers
  WHERE gst_verified = 1
)
SELECT category, name, city, ROUND(100.0 * response_rate, 0) AS response_pct, rating
FROM ranked
WHERE rn <= 2
ORDER BY category, rn;
