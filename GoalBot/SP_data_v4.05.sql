with city_details AS
(
select addressId, c.cityId, c.cityName
from address as a with (nolock)
inner join city AS c WITH (NOLOCK)
	ON a.cityId = c.cityId
)
--

SELECT 

		--sales
			s.id as salesId,
			s.invoiceId,
			--s.invoiceDate,
			  --Date
				CAST(s.invoiceDate AS DATE) AS invoiceDate,
				--DATENAME(QUARTER, s.invoiceDate) AS [quarter],
				--MONTH(s.invoiceDate) AS [month],
				--YEAR(s.invoiceDate) AS [year],
				--DATEPART(WEEKDAY, s.invoiceDate) AS [day_of_week],        -- 1 (Sunday) to 7 (Saturday) by default
				--DAY(s.invoiceDate) AS [day_of_month],
				--DATEPART(DAYOFYEAR, s.invoiceDate) AS [day_of_year],
				--DATEPART(WEEK, s.invoiceDate) AS [week_of_year],
			s.stockistId,
			s.chemistId,
			s.billAmount,

		--salesDetails
			sd.id as salesDetails_Id,
			sd.productCode,
			sd.productName,
			sd.productId,
			sd.mrp,
			sd.rate,
			sd.batch,
			sd.expiryDate,
			sd.qty,
			sd.free,
			sd.amount,
			sd.cashDiscountPercentage,
			sd.productDiscountPercentage,
			sd.productDiscount,
			sd.extraSchemePercentage,
			sd.extraScheme,
			sd.taxableAmount,
			sd.sgst,
			sd.sgstAmount,
			sd.cgst,
			sd.cgstAmount,
			sd.cess,
			sd.cessAmount,
			sd.ptr,
			sd.pts,
			sd.hsn,

		--products
			p.contents,
			p.packingTypeId,
			p.divisionId,

		--
			cd.cityId as cityId,
			hq.regionId,


		-- unified party info (stockist or chemist)
		CASE 
			WHEN s.stockistId IS NOT NULL THEN st.stockistName
			ELSE ch.chemistName
		END AS customerName,

		CASE 
			WHEN s.stockistId IS NOT NULL THEN st.addressId
			ELSE ch.addressId
		END AS addressId,

		CASE 
			WHEN s.stockistId IS NOT NULL THEN st.hqId
			ELSE ch.hqId
		END AS hqId,

		--chemist
			ch.categoryId,
			ch.chemistTypeId

		--hqCitymap
			--hcm.cityId,

		--hq
			--hq.regionId




		FROM sales s with (nolock)


		INNER JOIN salesDetails sd with (nolock)
			ON s.id = sd.salesId
 

	    LEFT JOIN products p with (nolock)
			ON sd.productId = p.productId

		LEFT JOIN stockist st with(nolock)
			ON s.stockistId = st.stockistId

		LEFT JOIN chemist ch  with(nolock)
			ON s.chemistId = ch.chemistId

		left join city_details as cd 
			on cd.addressId = CONCAT(ch.addressId, st.addressId)---

		--LEFT JOIN hqCitymap hcm with(nolock)
		--	ON st.hqId = hcm.hqId

		LEFT JOIN hq with(nolock)
			ON st.hqId = hq.hqId


		WHERE ( (s.stockistId IS NOT NULL and s.chemistId IS NULL) OR
		(s.stockistId IS NULL and s.chemistId IS NOT NULL) )
		AND p.isActive = 1;