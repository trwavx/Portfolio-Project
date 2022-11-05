create database covid;

use covid;

select *
from covid.covid_deaths
order by 3, 4;

-- Death percentage
select location, date, total_cases, total_deaths, (total_deaths / total_cases) * 100 as death_percentage
from covid.covid_deaths
order by 1, 2;

-- Countries with highest death count
select location, max(total_deaths) as highest_death_count
from covid.covid_deaths
group by location
order by highest_death_count desc;

-- Continent with highest death count
select continent, max(total_deaths) as highest_death_count
from covid.covid_deaths
group by continent
order by highest_death_count desc;

-- Percentage of population Infected with COVID
select location, date, population, total_cases, (total_cases / population) * 100 as percentage_of_population_infected_with_covid
from covid.covid_deaths
order by 1, 2;

-- Countries with highest percentage of population Infected with COVID
select location, population, max(total_cases) as highest_infection_count, max((total_cases / population)) * 100 as percentage_of_population_infected_with_covid
from covid.covid_deaths
group by location, population
order by percentage_of_population_infected_with_covid desc;

-- Global numbers
select sum(new_cases) as total_cases, sum(new_deaths) as total_deaths, sum(new_deaths) / sum(new_cases) * 100 as death_percentage
from covid.covid_deaths
order by 1, 2;

-- Percentage of population that has received at least one covid vaccine
with population_vs_vacc (continent, location, date, population, new_vaccinations, vaccinations_summary_by_countries)
as (
	select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations
	, sum(vac.new_vaccinations) over (partition by dea.location order by dea.location, dea.date) as vaccinations_summary_by_countries
	from covid.covid_deaths dea
	join covid.covid_vaccinations vac
		on dea.location = vac.location
		and dea.date = vac.date
)
select *, round((vaccinations_summary_by_countries / population) * 100, 2) as percentage_of_population_that_has_received_at_least_one_covid_vaccine
from population_vs_vacc;